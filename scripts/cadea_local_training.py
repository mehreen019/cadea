"""
CADEA Training Script - Optimized for Local Machine (3060 Ti - 8GB VRAM)
=========================================================================

Key optimizations:
- Batch size = 1 (fits in 8GB)
- Gradient checkpointing enabled
- Frequent checkpoints (every 250 steps)
- Memory monitoring
- Clean error handling

Usage:
    python cadea_local_training.py --test          # Quick 5-min test
    python cadea_local_training.py --stage en      # Just English
    python cadea_local_training.py --full          # Full experiment
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
import gc
import warnings
import json
import os
import argparse
import random
from pathlib import Path
from datetime import datetime
warnings.filterwarnings('ignore')
from dotenv import load_dotenv


def set_seed(seed: int):
    """Set random seed for reproducibility across all libraries"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

load_dotenv()

# ============================================================================
# CONFIGURATION FOR 3060 TI
# ============================================================================
class Config:
    """Optimized config for 8GB VRAM"""
    
    # Model
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    
    # Datasets
    en_dataset = "tatsu-lab/alpaca"
    en_samples = 100
    bn_dataset = "md-nishat-008/Bangla-Instruct"
    bn_samples = 2000  # Reduced for 3060 Ti
    ar_dataset = "arbml/CIDAR"
    ar_samples = 2000  # Reduced for 3060 Ti
    
    # Memory-optimized training params
    batch_size = 1  # CRITICAL for 8GB VRAM
    gradient_accumulation = 4  # Effective batch size = 4
    learning_rate = 5e-6
    
    # Monitoring
    layers_to_track = [0, 3, 6, 9, 12, 15]
    log_interval = 50
    checkpoint_interval = 250  # Frequent saves
    vram_log_interval = 100  # Log VRAM usage
    total_steps = 1500  # Reduced for faster iteration
    
    max_length = 128  # Reduced from 256 to prevent OOM
    use_bf16 = True  # 3060 Ti supports bfloat16
    gradient_checkpointing = True  # ESSENTIAL
    
    # Paths
    checkpoint_dir = "./checkpoints"
    results_dir = "./results"
    
    # HF Token (set via environment or here)
    hf_token = os.getenv("HF_TOKEN", "")

class TestConfig(Config):
    """Ultra-fast test config (5 minutes)"""
    en_samples = 10
    bn_samples = 50
    ar_samples = 50
    total_steps = 100
    checkpoint_interval = 50
    max_length = 128  # Reduced for OOM prevention

# ============================================================================
# CHECKPOINT MANAGER (Same as before but with local paths)
# ============================================================================
class CheckpointManager:
    def __init__(self, checkpoint_dir="./checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Checkpoint directory: {self.checkpoint_dir.absolute()}")
        
    def save_checkpoint(self, stage, step, model, optimizer, scheduler=None, 
                       metrics=None, conflict_history=None, training_metrics=None, 
                       performance_tracking=None, baseline_grads=None):
        """Save complete training state"""
        checkpoint_path = self.checkpoint_dir / f"{stage}_step_{step}.pt"
        
        print(f"\n💾 Saving checkpoint: {checkpoint_path.name}")
        
        checkpoint = {
            'stage': stage,
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics or {},
            'conflict_history': self._serialize_conflict_history(conflict_history or {}),
            'training_metrics': training_metrics or {},
            'performance_tracking': performance_tracking or {},
            'baseline_grads': baseline_grads
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        # Save metadata
        metadata = {
            'stage': stage,
            'step': step,
            'checkpoint_file': str(checkpoint_path),
            'timestamp': checkpoint['timestamp']
        }
        
        with open(self.checkpoint_dir / f"{stage}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update latest
        with open(self.checkpoint_dir / "latest.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        size_mb = checkpoint_path.stat().st_size / 1e6
        print(f"✓ Checkpoint saved: {size_mb:.1f} MB")
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path=None, model=None, optimizer=None, scheduler=None):
        """Load training state"""
        if checkpoint_path is None:
            latest_path = self.checkpoint_dir / "latest.json"
            if latest_path.exists():
                with open(latest_path, 'r') as f:
                    metadata = json.load(f)
                checkpoint_path = Path(metadata['checkpoint_file'])
            else:
                print("⚠️  No checkpoint found")
                return None
        
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            print(f"⚠️  Checkpoint not found: {checkpoint_path}")
            return None
        
        print(f"\n📂 Loading checkpoint: {checkpoint_path.name}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cuda')
        
        if model:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✓ Model state loaded")
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("✓ Optimizer state loaded")
        
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("✓ Scheduler state loaded")
        
        print(f"✓ Loaded from stage '{checkpoint['stage']}' at step {checkpoint['step']}")
        print(f"  Saved at: {checkpoint.get('timestamp', 'unknown')}")
        
        return {
            'stage': checkpoint['stage'],
            'step': checkpoint['step'],
            'metrics': checkpoint.get('metrics', {}),
            'conflict_history': self._deserialize_conflict_history(checkpoint.get('conflict_history', {})),
            'training_metrics': checkpoint.get('training_metrics', {}),
            'performance_tracking': checkpoint.get('performance_tracking', {}),
            'baseline_grads': checkpoint.get('baseline_grads')
        }
    
    def list_checkpoints(self):
        """List all checkpoints"""
        checkpoints = sorted(self.checkpoint_dir.glob("*.pt"))
        print(f"\n📋 Available checkpoints ({len(checkpoints)}):")
        for ckpt in checkpoints:
            size_mb = ckpt.stat().st_size / 1e6
            print(f"  • {ckpt.name} ({size_mb:.1f} MB)")
        return checkpoints
    
    def _serialize_conflict_history(self, conflict_history):
        if not conflict_history:
            return {}
        serialized = {}
        for layer, history in conflict_history.items():
            serialized[str(layer)] = [
                {k: float(v) if isinstance(v, (int, float, np.number, torch.Tensor)) else int(v) 
                 for k, v in h.items()}
                for h in history
            ]
        return serialized
    
    def _deserialize_conflict_history(self, serialized):
        if not serialized:
            return {}
        return {int(k): v for k, v in serialized.items()}

# ============================================================================
# VRAM MONITOR
# ============================================================================
class VRAMMonitor:
    """Monitor and log VRAM usage"""
    
    def __init__(self):
        self.peak_allocated = 0
        self.peak_reserved = 0
        
    def log(self, step=None):
        if not torch.cuda.is_available():
            return
        
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        
        self.peak_allocated = max(self.peak_allocated, allocated)
        self.peak_reserved = max(self.peak_reserved, reserved)
        
        prefix = f"Step {step} | " if step is not None else ""
        print(f"{prefix}VRAM: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
        
        # Warn if close to limit
        if allocated > 7.0:
            print("⚠️  WARNING: VRAM usage high (>7 GB), may OOM soon!")
        
        return allocated, reserved
    
    def summary(self):
        print(f"\n📊 VRAM Summary:")
        print(f"  Peak allocated: {self.peak_allocated:.2f} GB")
        print(f"  Peak reserved: {self.peak_reserved:.2f} GB")

# ============================================================================
# HELPER FUNCTIONS (Same as before)
# ============================================================================
def prepare_dataset(dataset_name, num_samples, max_length, tokenizer):
    """Load and prepare dataset"""
    try:
        if dataset_name == "tatsu-lab/alpaca":
            dataset = load_dataset(dataset_name, split=f"train[:{num_samples}]", trust_remote_code=True)
            def format_alpaca(example):
                if example.get("input", "").strip():
                    text = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
                else:
                    text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
                return {"text": text}
            dataset = dataset.map(format_alpaca, remove_columns=dataset.column_names)
            
        elif dataset_name == "md-nishat-008/Bangla-Instruct":
            dataset = load_dataset(dataset_name, split=f"train[:{num_samples}]", trust_remote_code=True)
            def format_bangla(example):
                instruction = example.get('instruction', '').strip()
                response = example.get('response', '').strip()
                if not instruction or not response:
                    return {"text": "", "valid": False}
                return {"text": f"### Instruction:\n{instruction}\n\n### Response:\n{response}", "valid": True}
            dataset = dataset.map(format_bangla, remove_columns=dataset.column_names)
            dataset = dataset.filter(lambda x: x["valid"]).remove_columns(["valid"])
            
        elif dataset_name == "arbml/CIDAR":
            actual_samples = min(num_samples, 10000)
            dataset = load_dataset(dataset_name, split=f"train[:{actual_samples}]", trust_remote_code=True)
            def format_arabic(example):
                instruction = example.get('instruction', '').strip()
                output = example.get('output', '').strip()
                if not instruction or not output:
                    return {"text": "", "valid": False}
                return {"text": f"### Instruction:\n{instruction}\n\n### Response:\n{output}", "valid": True}
            dataset = dataset.map(format_arabic, remove_columns=dataset.column_names)
            dataset = dataset.filter(lambda x: x["valid"]).remove_columns(["valid"])
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, max_length=max_length, 
                           padding="max_length", return_tensors="pt")
        
        dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"], desc="Tokenizing")
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        
        print(f"✓ Dataset prepared: {len(dataset)} samples")
        return dataset
        
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        raise

def evaluate_on_test(model, test_dataloader, language_name, device, tokenizer):
    """Evaluate with FIXED causal LM loss"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc=f"Eval {language_name}", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            labels = input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100
            labels[:, 0] = -100
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            loss = outputs.loss
            num_tokens = (labels != -100).sum().item()
            
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = torch.exp(torch.tensor(avg_loss))
    
    model.train()
    return {"loss": avg_loss, "perplexity": perplexity.item()}

def extract_layer_gradients(model, batch, layers_to_track, device, tokenizer):
    """Extract gradients"""
    model.zero_grad()
    
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    
    if torch.isnan(loss) or torch.isinf(loss):
        return None, loss.item()
    
    loss.backward()
    
    layer_grads = {layer: [] for layer in layers_to_track}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if "model.layers." in name and ".mlp." in name:
            try:
                layer_num = int(name.split("model.layers.")[1].split(".")[0])
                if layer_num in layers_to_track:
                    layer_grads[layer_num].append(param.grad.detach().clone().flatten())
            except (IndexError, ValueError):
                continue
    
    aggregated_grads = {}
    for layer, grads in layer_grads.items():
        if grads:
            aggregated_grads[layer] = torch.cat(grads)
        else:
            aggregated_grads[layer] = torch.zeros(1, device=device)
    
    return aggregated_grads, loss.item()

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================
def main(args):
    # Select config
    if args.test:
        config = TestConfig()
        print("🧪 Using TEST configuration (5-minute run)")
    else:
        config = Config()
        print("🚀 Using FULL configuration")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*80}")
    print(f"CADEA Training - Local Machine")
    print(f"{'='*80}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"{'='*80}\n")
    
    # Initialize monitors
    checkpoint_manager = CheckpointManager(config.checkpoint_dir)
    vram_monitor = VRAMMonitor()
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, trust_remote_code=True, token=config.hf_token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16 if config.use_bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        token=config.hf_token,
        attn_implementation="sdpa"
    )
    
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("✓ Gradient checkpointing enabled")
    
    model.train()
    print(f"✓ Model loaded: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params")
    
    vram_monitor.log()
    
    # Load datasets
    print(f"\n{'='*80}")
    print("PREPARING DATASETS")
    print(f"{'='*80}")
    
    en_dataset = prepare_dataset(config.en_dataset, config.en_samples, config.max_length, tokenizer)
    bn_dataset = prepare_dataset(config.bn_dataset, config.bn_samples, config.max_length, tokenizer)
    ar_dataset = prepare_dataset(config.ar_dataset, config.ar_samples, config.max_length, tokenizer)
    
    en_test = prepare_dataset(config.en_dataset, 50, config.max_length, tokenizer)
    bn_test = prepare_dataset(config.bn_dataset, 500, config.max_length, tokenizer)
    ar_test = prepare_dataset(config.ar_dataset, 500, config.max_length, tokenizer)
    
    en_dataloader = DataLoader(en_dataset, batch_size=config.batch_size, shuffle=False)
    bn_dataloader = DataLoader(bn_dataset, batch_size=config.batch_size, shuffle=True)
    ar_dataloader = DataLoader(ar_dataset, batch_size=config.batch_size, shuffle=True)
    
    en_test_loader = DataLoader(en_test, batch_size=config.batch_size, shuffle=False)
    bn_test_loader = DataLoader(bn_test, batch_size=config.batch_size, shuffle=False)
    ar_test_loader = DataLoader(ar_test, batch_size=config.batch_size, shuffle=False)
    
    # Initialize tracking
    performance_tracking = {"after_en": {}, "after_bn": {}, "after_ar": {}}
    
    # Check for resume
    checkpoint_manager.list_checkpoints()

    # Determine resume point
    resume_stage = None
    resume_step = 0
    latest_path = checkpoint_manager.checkpoint_dir / "latest.json"

    if latest_path.exists():
        with open(latest_path, 'r') as f:
            metadata = json.load(f)
        resume_stage = metadata.get('stage', '')
        resume_step = metadata.get('step', 0)
        print(f"\n📂 Found checkpoint: stage='{resume_stage}', step={resume_step}")

        # Determine which stages to skip
        if resume_stage in ['ar_complete']:
            print("✓ Training already complete! Nothing to do.")
            return
        elif resume_stage in ['bn_complete', 'ar']:
            print("✓ Skipping EN and BN stages (already complete)")
        elif resume_stage in ['en_complete', 'bn']:
            print("✓ Skipping EN stage (already complete)")

    # Stage 1: English (same as Kaggle version but with VRAM monitoring)
    skip_en = resume_stage in ['en_complete', 'bn', 'bn_complete', 'ar', 'ar_complete']

    if skip_en:
        print(f"\n{'='*80}")
        print("STAGE 1: ENGLISH TRAINING - SKIPPED (checkpoint found)")
        print(f"{'='*80}")
        # Load checkpoint to restore model state
        checkpoint_data = checkpoint_manager.load_checkpoint(model=model)
        if checkpoint_data:
            performance_tracking = checkpoint_data.get('performance_tracking', performance_tracking)
    else:
        print(f"\n{'='*80}")
        print("STAGE 1: ENGLISH TRAINING")
        print(f"{'='*80}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    if not skip_en:
        for step, batch in enumerate(tqdm(en_dataloader, desc="English")):
            try:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = input_ids.clone()
                labels[labels == tokenizer.pad_token_id] = -100

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"⚠️  NaN loss at step {step}, skipping")
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                # CRITICAL: Aggressive memory cleanup
                del input_ids, attention_mask, labels, outputs, loss

                if step % 10 == 0:
                    torch.cuda.empty_cache()

                if step % 10 == 0:
                    allocated = torch.cuda.memory_allocated() / 1e9
                    print(f"Step {step}, VRAM: {allocated:.2f}GB")

                if step % config.vram_log_interval == 0:
                    vram_monitor.log(step)

                if step > 0 and step % config.checkpoint_interval == 0:
                    checkpoint_manager.save_checkpoint(
                        stage='en', step=step, model=model, optimizer=optimizer,
                        performance_tracking=performance_tracking
                    )
                    # Clear cache after checkpoint save
                    torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n🚨 OOM at step {step}! Attempting recovery...")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    raise

        print("✓ English training complete")

    if not skip_en:
        # Evaluate after English
        print("\n" + "="*80)
        print("EVALUATION AFTER ENGLISH TRAINING")
        print("="*80)

        performance_tracking["after_en"]["en"] = evaluate_on_test(
            model, en_test_loader, "English", device, tokenizer
        )
        print(f"EN Test - Loss: {performance_tracking['after_en']['en']['loss']:.4f}, "
              f"Perplexity: {performance_tracking['after_en']['en']['perplexity']:.2f}")

        # Save post-English checkpoint
        checkpoint_manager.save_checkpoint(
            stage='en_complete',
            step=len(en_dataloader),
            model=model,
            optimizer=optimizer,
            scheduler=None,
            metrics={},
            conflict_history={},
            training_metrics={},
            performance_tracking=performance_tracking
        )

        # Compute English baseline gradients
        print("\n" + "="*80)
        print("COMPUTING ENGLISH BASELINE GRADIENT PROFILE")
        print("="*80)

        english_baseline_grads = {layer: None for layer in config.layers_to_track}
        en_losses = []

        num_baseline_batches = min(20, len(en_dataloader))

        for i, batch in enumerate(tqdm(en_dataloader, total=num_baseline_batches)):
            if i >= num_baseline_batches:
                break

            layer_grads, loss = extract_layer_gradients(model, batch, config.layers_to_track, device, tokenizer)
            en_losses.append(loss)

            for layer, grad in layer_grads.items():
                if english_baseline_grads[layer] is None:
                    english_baseline_grads[layer] = grad.clone()
                else:
                    english_baseline_grads[layer] += grad

            model.zero_grad()

            if i % 10 == 0:
                torch.cuda.empty_cache()

        # Average
        for layer in english_baseline_grads:
            if english_baseline_grads[layer] is not None:
                english_baseline_grads[layer] /= num_baseline_batches

        print(f"\n✓ English baseline computed")
        print(f"  Average loss: {np.mean(en_losses):.4f}")

        # Save baseline gradients
        baseline_path = checkpoint_manager.checkpoint_dir / "english_baseline_grads.pt"
        torch.save(english_baseline_grads, baseline_path)
        print(f"✓ Baseline gradients saved: {baseline_path}")
    else:
        # Load English baseline gradients from file
        baseline_path = checkpoint_manager.checkpoint_dir / "english_baseline_grads.pt"
        if baseline_path.exists():
            english_baseline_grads = torch.load(baseline_path, map_location=device)
            print(f"✓ Loaded English baseline gradients from: {baseline_path}")
        else:
            print("⚠️  English baseline gradients not found, will recompute...")
            # Recompute if missing
            english_baseline_grads = {layer: None for layer in config.layers_to_track}
            num_baseline_batches = min(20, len(en_dataloader))
            for i, batch in enumerate(tqdm(en_dataloader, total=num_baseline_batches, desc="Recomputing EN baseline")):
                if i >= num_baseline_batches:
                    break
                layer_grads, _ = extract_layer_gradients(model, batch, config.layers_to_track, device, tokenizer)
                for layer, grad in layer_grads.items():
                    if english_baseline_grads[layer] is None:
                        english_baseline_grads[layer] = grad.clone()
                    else:
                        english_baseline_grads[layer] += grad
                model.zero_grad()
            for layer in english_baseline_grads:
                if english_baseline_grads[layer] is not None:
                    english_baseline_grads[layer] /= num_baseline_batches
            torch.save(english_baseline_grads, baseline_path)
    
    # ========================================================================
    # STAGE 2: BENGALI TRAINING WITH CONFLICT MONITORING
    # ========================================================================
    skip_bn = resume_stage in ['bn_complete', 'ar', 'ar_complete']

    if skip_bn:
        print(f"\n{'='*80}")
        print("STAGE 2: BENGALI TRAINING - SKIPPED (checkpoint found)")
        print(f"{'='*80}")
    else:
        print(f"\n{'='*80}")
        print("STAGE 2: BENGALI TRAINING WITH CONFLICT MONITORING (vs English baseline)")
        print(f"{'='*80}")

    # Initialize tracking structures (needed even when skipping for later stages)
    conflict_history_bn = {layer: [] for layer in config.layers_to_track}
    training_metrics_bn = {
        'steps': [],
        'losses': [],
        'peak_conflict_layers': []
    }

    if not skip_bn:
        # Setup optimizer with scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=50,
            num_training_steps=config.total_steps
        )

        # Training loop
        bn_dataloader_iter = iter(bn_dataloader)
        step = 0
        accumulated_loss = 0

        print(f"Training on Bengali for {config.total_steps} steps (logging every {config.log_interval})...")

        progress_bar = tqdm(total=config.total_steps, desc="Bengali training")

        while step < config.total_steps:
            # Get next batch (cycle through dataset if needed)
            try:
                batch = next(bn_dataloader_iter)
            except StopIteration:
                bn_dataloader_iter = iter(bn_dataloader)
                batch = next(bn_dataloader_iter)

            # Forward and backward
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss / config.gradient_accumulation

            # Check for NaN loss before backward
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\nWARNING: NaN loss at step {step}, skipping batch")
                step += 1
                progress_bar.update(1)
                continue

            loss.backward()
            accumulated_loss += loss.item()

            # === CONFLICT MONITORING (extract gradients from CURRENT backward) ===
            if step % config.log_interval == 0:
                # Extract gradients directly from current backward pass
                layer_grads = {}
                for name, param in model.named_parameters():
                    if param.grad is None:
                        continue

                    if ".layers." in name and ".mlp." in name:
                        try:
                            parts = name.split(".layers.")[1].split(".")
                            layer_num = int(parts[0])

                            if layer_num in config.layers_to_track:
                                if layer_num not in layer_grads:
                                    layer_grads[layer_num] = []
                                layer_grads[layer_num].append(param.grad.detach().clone().flatten())
                        except (IndexError, ValueError):
                            continue

                # Concatenate gradients per layer
                bn_layer_grads = {}
                for layer, grads in layer_grads.items():
                    if grads:
                        bn_layer_grads[layer] = torch.cat(grads)

                # Compute cosine similarity with English baseline
                layer_conflicts = {}
                for layer in config.layers_to_track:
                    if layer in bn_layer_grads and english_baseline_grads[layer] is not None:
                        # Normalize gradients
                        en_grad_norm = F.normalize(english_baseline_grads[layer].unsqueeze(0), dim=1)
                        bn_grad_norm = F.normalize(bn_layer_grads[layer].unsqueeze(0), dim=1)

                        # Cosine similarity
                        cos_sim = F.cosine_similarity(en_grad_norm, bn_grad_norm).item()

                        # Store conflict metrics
                        conflict_history_bn[layer].append({
                            'step': step,
                            'cosine_similarity': cos_sim,
                            'conflict_score': 1 - cos_sim,
                            'bn_grad_norm': bn_layer_grads[layer].norm().item(),
                            'en_grad_norm': english_baseline_grads[layer].norm().item()
                        })

                        layer_conflicts[layer] = 1 - cos_sim

                # Find peak conflict layer
                if layer_conflicts:
                    peak_conflict_layer = max(layer_conflicts, key=layer_conflicts.get)
                    training_metrics_bn['peak_conflict_layers'].append(peak_conflict_layer)
                    training_metrics_bn['steps'].append(step)
                    training_metrics_bn['losses'].append(accumulated_loss * config.gradient_accumulation)

                    accumulated_loss = 0

                    # Log
                    progress_bar.set_postfix({
                        'loss': f"{training_metrics_bn['losses'][-1]:.4f}",
                        'peak_layer': peak_conflict_layer,
                        'max_conflict': f"{max(layer_conflicts.values()):.3f}"
                    })

            # Optimizer step every gradient_accumulation steps
            if (step + 1) % config.gradient_accumulation == 0:
                # Gradient clipping
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                if torch.isnan(total_norm) or torch.isinf(total_norm):
                    print(f"\nWARNING: NaN gradients at step {step}, skipping update")
                    optimizer.zero_grad()
                else:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            step += 1
            progress_bar.update(1)

            # Memory management and VRAM logging
            if step % config.vram_log_interval == 0:
                vram_monitor.log(step)

            if step % 100 == 0:
                torch.cuda.empty_cache()

            # Checkpoint saving
            if step > 0 and step % config.checkpoint_interval == 0:
                checkpoint_manager.save_checkpoint(
                    stage='bn', step=step, model=model, optimizer=optimizer,
                    scheduler=scheduler, conflict_history=conflict_history_bn,
                    training_metrics=training_metrics_bn,
                    performance_tracking=performance_tracking,
                    baseline_grads=english_baseline_grads
                )

        progress_bar.close()
        print("\n✓ Bengali training complete")

        # Evaluate after Bengali
        print("\n" + "="*80)
        print("EVALUATION AFTER BENGALI TRAINING")
        print("="*80)

        performance_tracking["after_bn"]["en"] = evaluate_on_test(
            model, en_test_loader, "English", device, tokenizer
        )
        performance_tracking["after_bn"]["bn"] = evaluate_on_test(
            model, bn_test_loader, "Bengali", device, tokenizer
        )

        print(f"EN Test - Loss: {performance_tracking['after_bn']['en']['loss']:.4f}, "
              f"Perplexity: {performance_tracking['after_bn']['en']['perplexity']:.2f}")
        print(f"BN Test - Loss: {performance_tracking['after_bn']['bn']['loss']:.4f}, "
              f"Perplexity: {performance_tracking['after_bn']['bn']['perplexity']:.2f}")

        en_degradation = ((performance_tracking['after_bn']['en']['perplexity'] -
                           performance_tracking['after_en']['en']['perplexity']) /
                          performance_tracking['after_en']['en']['perplexity']) * 100
        print(f"\n⚠️  EN Performance Degradation: {en_degradation:+.1f}%")

        # Save post-Bengali checkpoint
        checkpoint_manager.save_checkpoint(
            stage='bn_complete', step=config.total_steps, model=model, optimizer=optimizer,
            scheduler=scheduler, conflict_history=conflict_history_bn,
            training_metrics=training_metrics_bn,
            performance_tracking=performance_tracking,
            baseline_grads=english_baseline_grads
        )

        # ========================================================================
        # COMPUTE BENGALI BASELINE GRADIENT PROFILE
        # ========================================================================
        print("\n" + "="*80)
        print("COMPUTING BENGALI BASELINE GRADIENT PROFILE (post-training)")
        print("="*80)

        bengali_baseline_grads = {layer: None for layer in config.layers_to_track}
        bn_baseline_losses = []

        num_baseline_batches = min(20, len(bn_dataloader))

        for i, batch in enumerate(tqdm(bn_dataloader, total=num_baseline_batches, desc="Bengali baseline")):
            if i >= num_baseline_batches:
                break

            layer_grads, loss = extract_layer_gradients(model, batch, config.layers_to_track, device, tokenizer)
            bn_baseline_losses.append(loss)

            for layer, grad in layer_grads.items():
                if bengali_baseline_grads[layer] is None:
                    bengali_baseline_grads[layer] = grad.clone()
                else:
                    bengali_baseline_grads[layer] += grad

            model.zero_grad()

            if i % 10 == 0:
                torch.cuda.empty_cache()

        # Average
        for layer in bengali_baseline_grads:
            if bengali_baseline_grads[layer] is not None:
                bengali_baseline_grads[layer] /= num_baseline_batches

        print(f"\n✓ Bengali baseline computed")
        print(f"  Average loss: {np.mean(bn_baseline_losses):.4f}")

        # Save Bengali baseline gradients
        bn_baseline_path = checkpoint_manager.checkpoint_dir / "bengali_baseline_grads.pt"
        torch.save(bengali_baseline_grads, bn_baseline_path)
        print(f"✓ Bengali baseline gradients saved: {bn_baseline_path}")
    else:
        # Load Bengali baseline gradients from file
        bn_baseline_path = checkpoint_manager.checkpoint_dir / "bengali_baseline_grads.pt"
        if bn_baseline_path.exists():
            bengali_baseline_grads = torch.load(bn_baseline_path, map_location=device)
            print(f"✓ Loaded Bengali baseline gradients from: {bn_baseline_path}")
        else:
            print("⚠️  Bengali baseline gradients not found, will recompute...")
            bengali_baseline_grads = {layer: None for layer in config.layers_to_track}
            num_baseline_batches = min(20, len(bn_dataloader))
            for i, batch in enumerate(tqdm(bn_dataloader, total=num_baseline_batches, desc="Recomputing BN baseline")):
                if i >= num_baseline_batches:
                    break
                layer_grads, _ = extract_layer_gradients(model, batch, config.layers_to_track, device, tokenizer)
                for layer, grad in layer_grads.items():
                    if bengali_baseline_grads[layer] is None:
                        bengali_baseline_grads[layer] = grad.clone()
                    else:
                        bengali_baseline_grads[layer] += grad
                model.zero_grad()
            for layer in bengali_baseline_grads:
                if bengali_baseline_grads[layer] is not None:
                    bengali_baseline_grads[layer] /= num_baseline_batches
            torch.save(bengali_baseline_grads, bn_baseline_path)

    # ========================================================================
    # STAGE 3: ARABIC TRAINING WITH CONFLICT MONITORING
    # ========================================================================
    print(f"\n{'='*80}")
    print("STAGE 3: ARABIC TRAINING WITH CONFLICT MONITORING (vs Bengali baseline)")
    print(f"{'='*80}")

    # RE-INITIALIZE optimizer (fresh training phase)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=50,
        num_training_steps=config.total_steps
    )

    # Tracking structures
    conflict_history_ar = {layer: [] for layer in config.layers_to_track}
    training_metrics_ar = {
        'steps': [],
        'losses': [],
        'peak_conflict_layers': []
    }

    # Training loop
    ar_dataloader_iter = iter(ar_dataloader)
    step = 0
    accumulated_loss = 0

    print(f"Training on Arabic for {config.total_steps} steps (logging every {config.log_interval})...")

    progress_bar = tqdm(total=config.total_steps, desc="Arabic training")

    while step < config.total_steps:
        # Get next batch (cycle through dataset if needed)
        try:
            batch = next(ar_dataloader_iter)
        except StopIteration:
            ar_dataloader_iter = iter(ar_dataloader)
            batch = next(ar_dataloader_iter)

        # Forward and backward
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss / config.gradient_accumulation

        # Check for NaN loss before backward
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\nWARNING: NaN loss at step {step}, skipping batch")
            step += 1
            progress_bar.update(1)
            continue

        loss.backward()
        accumulated_loss += loss.item()

        # === CONFLICT MONITORING (extract gradients from CURRENT backward) ===
        if step % config.log_interval == 0:
            # Extract gradients directly from current backward pass
            layer_grads = {}
            for name, param in model.named_parameters():
                if param.grad is None:
                    continue

                if ".layers." in name and ".mlp." in name:
                    try:
                        parts = name.split(".layers.")[1].split(".")
                        layer_num = int(parts[0])

                        if layer_num in config.layers_to_track:
                            if layer_num not in layer_grads:
                                layer_grads[layer_num] = []
                            layer_grads[layer_num].append(param.grad.detach().clone().flatten())
                    except (IndexError, ValueError):
                        continue

            # Concatenate gradients per layer
            ar_layer_grads = {}
            for layer, grads in layer_grads.items():
                if grads:
                    ar_layer_grads[layer] = torch.cat(grads)

            # Compute cosine similarity with BENGALI baseline
            layer_conflicts = {}
            for layer in config.layers_to_track:
                if layer in ar_layer_grads and bengali_baseline_grads[layer] is not None:
                    # Normalize gradients
                    bn_grad_norm = F.normalize(bengali_baseline_grads[layer].unsqueeze(0), dim=1)
                    ar_grad_norm = F.normalize(ar_layer_grads[layer].unsqueeze(0), dim=1)

                    # Cosine similarity
                    cos_sim = F.cosine_similarity(bn_grad_norm, ar_grad_norm).item()

                    # Store conflict metrics
                    conflict_history_ar[layer].append({
                        'step': step,
                        'cosine_similarity': cos_sim,
                        'conflict_score': 1 - cos_sim,
                        'ar_grad_norm': ar_layer_grads[layer].norm().item(),
                        'bn_grad_norm': bengali_baseline_grads[layer].norm().item()
                    })

                    layer_conflicts[layer] = 1 - cos_sim

            # Find peak conflict layer
            if layer_conflicts:
                peak_conflict_layer = max(layer_conflicts, key=layer_conflicts.get)
                training_metrics_ar['peak_conflict_layers'].append(peak_conflict_layer)
                training_metrics_ar['steps'].append(step)
                training_metrics_ar['losses'].append(accumulated_loss * config.gradient_accumulation)

                accumulated_loss = 0

                # Log
                progress_bar.set_postfix({
                    'loss': f"{training_metrics_ar['losses'][-1]:.4f}",
                    'peak_layer': peak_conflict_layer,
                    'max_conflict': f"{max(layer_conflicts.values()):.3f}"
                })

        # Optimizer step every gradient_accumulation steps
        if (step + 1) % config.gradient_accumulation == 0:
            # Gradient clipping
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if torch.isnan(total_norm) or torch.isinf(total_norm):
                print(f"\nWARNING: NaN gradients at step {step}, skipping update")
                optimizer.zero_grad()
            else:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        step += 1
        progress_bar.update(1)

        # Memory management and VRAM logging
        if step % config.vram_log_interval == 0:
            vram_monitor.log(step)

        if step % 100 == 0:
            torch.cuda.empty_cache()

        # Checkpoint saving
        if step > 0 and step % config.checkpoint_interval == 0:
            checkpoint_manager.save_checkpoint(
                stage='ar', step=step, model=model, optimizer=optimizer,
                scheduler=scheduler, conflict_history=conflict_history_ar,
                training_metrics=training_metrics_ar,
                performance_tracking=performance_tracking
            )

    progress_bar.close()
    print("\n✓ Arabic training complete")

    # Evaluate after Arabic
    print("\n" + "="*80)
    print("EVALUATION AFTER ARABIC TRAINING")
    print("="*80)

    performance_tracking["after_ar"]["en"] = evaluate_on_test(
        model, en_test_loader, "English", device, tokenizer
    )
    performance_tracking["after_ar"]["bn"] = evaluate_on_test(
        model, bn_test_loader, "Bengali", device, tokenizer
    )
    performance_tracking["after_ar"]["ar"] = evaluate_on_test(
        model, ar_test_loader, "Arabic", device, tokenizer
    )

    print(f"EN Test - Loss: {performance_tracking['after_ar']['en']['loss']:.4f}, "
          f"Perplexity: {performance_tracking['after_ar']['en']['perplexity']:.2f}")
    print(f"BN Test - Loss: {performance_tracking['after_ar']['bn']['loss']:.4f}, "
          f"Perplexity: {performance_tracking['after_ar']['bn']['perplexity']:.2f}")
    print(f"AR Test - Loss: {performance_tracking['after_ar']['ar']['loss']:.4f}, "
          f"Perplexity: {performance_tracking['after_ar']['ar']['perplexity']:.2f}")

    # Save final checkpoint
    checkpoint_manager.save_checkpoint(
        stage='ar_complete', step=config.total_steps, model=model, optimizer=optimizer,
        scheduler=scheduler, conflict_history=conflict_history_ar,
        training_metrics=training_metrics_ar,
        performance_tracking=performance_tracking
    )

    # ========================================================================
    # COMBINED VISUALIZATION
    # ========================================================================
    print("\n" + "="*80)
    print("GENERATING COMBINED ANALYSIS VISUALIZATION")
    print("="*80)

    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    fig.suptitle('Sequential Transfer Analysis: EN→BN→AR\nNon-Stationary Gradient Conflict Migration',
                 fontsize=18, fontweight='bold', y=0.98)

    # Row 1: Conflict Evolution by Layer
    ax1 = fig.add_subplot(gs[0, 0])
    for layer in config.layers_to_track:
        steps = [h['step'] for h in conflict_history_bn[layer]]
        conflicts = [h['conflict_score'] for h in conflict_history_bn[layer]]
        ax1.plot(steps, conflicts, marker='o', markersize=4, label=f'Layer {layer}', linewidth=2, alpha=0.8)

    ax1.set_xlabel('Training Steps', fontsize=11)
    ax1.set_ylabel('Conflict Score (1 - cosine_sim)', fontsize=11)
    ax1.set_title('Stage 1: EN→BN Gradient Conflicts', fontsize=13, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, ncol=1)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.4, linewidth=1.5)
    ax1.set_ylim([0, 1])

    ax2 = fig.add_subplot(gs[0, 1])
    for layer in config.layers_to_track:
        steps = [h['step'] for h in conflict_history_ar[layer]]
        conflicts = [h['conflict_score'] for h in conflict_history_ar[layer]]
        ax2.plot(steps, conflicts, marker='s', markersize=4, label=f'Layer {layer}', linewidth=2, alpha=0.8)

    ax2.set_xlabel('Training Steps', fontsize=11)
    ax2.set_ylabel('Conflict Score (1 - cosine_sim)', fontsize=11)
    ax2.set_title('Stage 2: BN→AR Gradient Conflicts', fontsize=13, fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, ncol=1)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.4, linewidth=1.5)
    ax2.set_ylim([0, 1])

    # Final conflicts comparison
    ax3 = fig.add_subplot(gs[0, 2])
    final_conflicts_bn = {}
    final_conflicts_ar = {}

    for layer in config.layers_to_track:
        if conflict_history_bn[layer]:
            final_conflicts_bn[layer] = conflict_history_bn[layer][-1]['conflict_score']
        if conflict_history_ar[layer]:
            final_conflicts_ar[layer] = conflict_history_ar[layer][-1]['conflict_score']

    x = np.arange(len(config.layers_to_track))
    width = 0.35

    ax3.bar(x - width/2, [final_conflicts_bn.get(l, 0) for l in config.layers_to_track],
            width, label='EN→BN', color='steelblue', alpha=0.8)
    ax3.bar(x + width/2, [final_conflicts_ar.get(l, 0) for l in config.layers_to_track],
            width, label='BN→AR', color='coral', alpha=0.8)

    ax3.set_xlabel('Layer', fontsize=11)
    ax3.set_ylabel('Final Conflict Score', fontsize=11)
    ax3.set_title('Final Conflict Comparison', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'L{l}' for l in config.layers_to_track])
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')

    # Row 2: Peak Conflict Layer Migration
    ax4 = fig.add_subplot(gs[1, 0])
    checkpoint_indices_bn = list(range(len(training_metrics_bn['peak_conflict_layers'])))
    ax4.plot(checkpoint_indices_bn, training_metrics_bn['peak_conflict_layers'],
             marker='o', markersize=8, linewidth=2.5, color='darkblue', label='Peak Layer')
    ax4.fill_between(checkpoint_indices_bn, training_metrics_bn['peak_conflict_layers'],
                      alpha=0.2, color='blue')

    ax4.set_xlabel(f'Checkpoint (every {config.log_interval} steps)', fontsize=11)
    ax4.set_ylabel('Layer with Peak Conflict', fontsize=11)
    ax4.set_title('Stage 1: EN→BN Peak Migration', fontsize=13, fontweight='bold')
    ax4.set_yticks(config.layers_to_track)
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10)

    if training_metrics_bn['peak_conflict_layers']:
        initial_peak_bn = training_metrics_bn['peak_conflict_layers'][0]
        final_peak_bn = training_metrics_bn['peak_conflict_layers'][-1]
        ax4.axhline(y=initial_peak_bn, color='blue', linestyle='--', alpha=0.5, linewidth=2)
        ax4.axhline(y=final_peak_bn, color='green', linestyle='--', alpha=0.5, linewidth=2)
        ax4.text(0.02, 0.98, f'Initial: L{initial_peak_bn}\nFinal: L{final_peak_bn}',
                 transform=ax4.transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax5 = fig.add_subplot(gs[1, 1])
    checkpoint_indices_ar = list(range(len(training_metrics_ar['peak_conflict_layers'])))
    ax5.plot(checkpoint_indices_ar, training_metrics_ar['peak_conflict_layers'],
             marker='s', markersize=8, linewidth=2.5, color='darkred', label='Peak Layer')
    ax5.fill_between(checkpoint_indices_ar, training_metrics_ar['peak_conflict_layers'],
                      alpha=0.2, color='red')

    ax5.set_xlabel(f'Checkpoint (every {config.log_interval} steps)', fontsize=11)
    ax5.set_ylabel('Layer with Peak Conflict', fontsize=11)
    ax5.set_title('Stage 2: BN→AR Peak Migration', fontsize=13, fontweight='bold')
    ax5.set_yticks(config.layers_to_track)
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=10)

    if training_metrics_ar['peak_conflict_layers']:
        initial_peak_ar = training_metrics_ar['peak_conflict_layers'][0]
        final_peak_ar = training_metrics_ar['peak_conflict_layers'][-1]
        ax5.axhline(y=initial_peak_ar, color='blue', linestyle='--', alpha=0.5, linewidth=2)
        ax5.axhline(y=final_peak_ar, color='green', linestyle='--', alpha=0.5, linewidth=2)
        ax5.text(0.02, 0.98, f'Initial: L{initial_peak_ar}\nFinal: L{final_peak_ar}',
                 transform=ax5.transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Combined Peak Migration Journey
    ax6 = fig.add_subplot(gs[1, 2])
    total_checkpoints_bn = len(training_metrics_bn['peak_conflict_layers'])
    total_checkpoints_ar = len(training_metrics_ar['peak_conflict_layers'])
    timeline = list(range(total_checkpoints_bn + total_checkpoints_ar))

    combined_peaks = (training_metrics_bn['peak_conflict_layers'] +
                      training_metrics_ar['peak_conflict_layers'])

    if combined_peaks:
        ax6.plot(timeline[:total_checkpoints_bn], combined_peaks[:total_checkpoints_bn],
                 marker='o', markersize=6, linewidth=2.5, color='steelblue', label='EN→BN Stage')
        ax6.plot(timeline[total_checkpoints_bn-1:], combined_peaks[total_checkpoints_bn-1:],
                 marker='s', markersize=6, linewidth=2.5, color='coral', label='BN→AR Stage')
        ax6.axvline(x=total_checkpoints_bn, color='black', linestyle='--', linewidth=2,
                    alpha=0.7, label='Stage Transition')

    ax6.set_xlabel('Combined Timeline (checkpoints)', fontsize=11)
    ax6.set_ylabel('Layer with Peak Conflict', fontsize=11)
    ax6.set_title('Full Journey: Peak Conflict Migration', fontsize=13, fontweight='bold')
    ax6.set_yticks(config.layers_to_track)
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)

    # Row 3: Training Dynamics
    ax7 = fig.add_subplot(gs[2, 0])
    if training_metrics_bn['steps']:
        ax7.plot(training_metrics_bn['steps'], training_metrics_bn['losses'],
                 linewidth=2.5, color='steelblue', label='EN→BN', alpha=0.8)
    ax7_twin = ax7.twinx()
    if training_metrics_ar['steps']:
        ax7_twin.plot(training_metrics_ar['steps'], training_metrics_ar['losses'],
                      linewidth=2.5, color='coral', label='BN→AR', alpha=0.8)

    ax7.set_xlabel('Training Steps', fontsize=11)
    ax7.set_ylabel('EN→BN Loss', fontsize=11, color='steelblue')
    ax7_twin.set_ylabel('BN→AR Loss', fontsize=11, color='coral')
    ax7.set_title('Training Loss Trajectories', fontsize=13, fontweight='bold')
    ax7.tick_params(axis='y', labelcolor='steelblue')
    ax7_twin.tick_params(axis='y', labelcolor='coral')
    ax7.grid(True, alpha=0.3)

    lines1, labels1 = ax7.get_legend_handles_labels()
    lines2, labels2 = ax7_twin.get_legend_handles_labels()
    ax7.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)

    # Gradient Magnitude Heatmaps
    ax8 = fig.add_subplot(gs[2, 1])
    if conflict_history_bn[config.layers_to_track[0]]:
        grad_magnitudes_bn = np.zeros((len(config.layers_to_track), len(conflict_history_bn[config.layers_to_track[0]])))
        for i, layer in enumerate(config.layers_to_track):
            for j, hist in enumerate(conflict_history_bn[layer]):
                grad_magnitudes_bn[i, j] = hist['bn_grad_norm']

        im1 = ax8.imshow(grad_magnitudes_bn, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax8.set_yticks(range(len(config.layers_to_track)))
        ax8.set_yticklabels([f'L{l}' for l in config.layers_to_track])
        ax8.set_xlabel(f'Checkpoint (every {config.log_interval} steps)', fontsize=11)
        ax8.set_ylabel('Layer', fontsize=11)
        ax8.set_title('EN→BN Gradient Magnitude Heatmap', fontsize=13, fontweight='bold')
        plt.colorbar(im1, ax=ax8, label='Gradient Norm')

    ax9 = fig.add_subplot(gs[2, 2])
    if conflict_history_ar[config.layers_to_track[0]]:
        grad_magnitudes_ar = np.zeros((len(config.layers_to_track), len(conflict_history_ar[config.layers_to_track[0]])))
        for i, layer in enumerate(config.layers_to_track):
            for j, hist in enumerate(conflict_history_ar[layer]):
                grad_magnitudes_ar[i, j] = hist['ar_grad_norm']

        im2 = ax9.imshow(grad_magnitudes_ar, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax9.set_yticks(range(len(config.layers_to_track)))
        ax9.set_yticklabels([f'L{l}' for l in config.layers_to_track])
        ax9.set_xlabel(f'Checkpoint (every {config.log_interval} steps)', fontsize=11)
        ax9.set_ylabel('Layer', fontsize=11)
        ax9.set_title('BN→AR Gradient Magnitude Heatmap', fontsize=13, fontweight='bold')
        plt.colorbar(im2, ax=ax9, label='Gradient Norm')

    plot_path = results_dir / 'combined_en_bn_ar_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Combined visualization saved: {plot_path}")
    plt.close()

    # ========================================================================
    # VALIDATION METRICS
    # ========================================================================
    print("\n" + "="*80)
    print("VALIDATION RESULTS - SEQUENTIAL TRANSFER ANALYSIS")
    print("="*80)

    # Stage 1: EN→BN Analysis
    print("\n" + "="*80)
    print("STAGE 1: EN→BN VALIDATION")
    print("="*80)

    peak_layers_bn = training_metrics_bn['peak_conflict_layers']
    unique_peaks_bn = set(peak_layers_bn) if peak_layers_bn else set()
    migrations_bn = sum(1 for i in range(1, len(peak_layers_bn)) if peak_layers_bn[i] != peak_layers_bn[i-1]) if len(peak_layers_bn) > 1 else 0

    conflict_variance_bn = {}
    for layer in config.layers_to_track:
        conflicts = [h['conflict_score'] for h in conflict_history_bn[layer]]
        conflict_variance_bn[layer] = np.var(conflicts) if conflicts else 0

    high_variance_layers_bn = sorted(conflict_variance_bn.items(), key=lambda x: x[1], reverse=True)[:3]

    print(f"\n📊 KEY FINDINGS (EN→BN):")
    print(f"  ├─ Peak conflict migrated {migrations_bn} times across {len(peak_layers_bn)} checkpoints")
    print(f"  ├─ {len(unique_peaks_bn)} unique layers experienced peak conflict")
    if peak_layers_bn:
        print(f"  ├─ Initial peak: Layer {peak_layers_bn[0]}")
        print(f"  └─ Final peak: Layer {peak_layers_bn[-1]}")

    print(f"\n📈 MOST NON-STATIONARY LAYERS (EN→BN):")
    for layer, variance in high_variance_layers_bn:
        print(f"  ├─ Layer {layer}: variance = {variance:.4f}")

    # Hypothesis validation for EN→BN
    print(f"\n🎯 HYPOTHESIS VALIDATION (EN→BN):")
    hypothesis_validated_bn = False
    layer_distance_bn = 0

    if peak_layers_bn:
        layer_distance_bn = abs(peak_layers_bn[-1] - peak_layers_bn[0])
        if layer_distance_bn >= 4:
            print(f"  ✅ STRONG: Peak conflict migrated {layer_distance_bn} layers (shallow→deep or vice versa)")
            hypothesis_validated_bn = True
        elif migrations_bn >= 5:
            print(f"  ✅ MODERATE: Peak conflict oscillated across {migrations_bn} layer transitions")
            hypothesis_validated_bn = True
        else:
            print(f"  ⚠️  WEAK: Limited peak migration detected ({migrations_bn} transitions)")

    if conflict_variance_bn:
        max_variance_bn = max(conflict_variance_bn.values())
        min_variance_bn = min(v for v in conflict_variance_bn.values() if v > 0) if any(v > 0 for v in conflict_variance_bn.values()) else 1
        if min_variance_bn > 0 and max_variance_bn / min_variance_bn > 5:
            print(f"  ✅ Conflict variance highly non-uniform (ratio: {max_variance_bn/min_variance_bn:.2f}x)")
            hypothesis_validated_bn = True

    # Stage 2: BN→AR Analysis
    print("\n" + "="*80)
    print("STAGE 2: BN→AR VALIDATION")
    print("="*80)

    peak_layers_ar = training_metrics_ar['peak_conflict_layers']
    unique_peaks_ar = set(peak_layers_ar) if peak_layers_ar else set()
    migrations_ar = sum(1 for i in range(1, len(peak_layers_ar)) if peak_layers_ar[i] != peak_layers_ar[i-1]) if len(peak_layers_ar) > 1 else 0

    conflict_variance_ar = {}
    for layer in config.layers_to_track:
        conflicts = [h['conflict_score'] for h in conflict_history_ar[layer]]
        conflict_variance_ar[layer] = np.var(conflicts) if conflicts else 0

    high_variance_layers_ar = sorted(conflict_variance_ar.items(), key=lambda x: x[1], reverse=True)[:3]

    print(f"\n📊 KEY FINDINGS (BN→AR):")
    print(f"  ├─ Peak conflict migrated {migrations_ar} times across {len(peak_layers_ar)} checkpoints")
    print(f"  ├─ {len(unique_peaks_ar)} unique layers experienced peak conflict")
    if peak_layers_ar:
        print(f"  ├─ Initial peak: Layer {peak_layers_ar[0]}")
        print(f"  └─ Final peak: Layer {peak_layers_ar[-1]}")

    print(f"\n📈 MOST NON-STATIONARY LAYERS (BN→AR):")
    for layer, variance in high_variance_layers_ar:
        print(f"  ├─ Layer {layer}: variance = {variance:.4f}")

    # Hypothesis validation for BN→AR
    print(f"\n🎯 HYPOTHESIS VALIDATION (BN→AR):")
    hypothesis_validated_ar = False
    layer_distance_ar = 0

    if peak_layers_ar:
        layer_distance_ar = abs(peak_layers_ar[-1] - peak_layers_ar[0])
        if layer_distance_ar >= 4:
            print(f"  ✅ STRONG: Peak conflict migrated {layer_distance_ar} layers (shallow→deep or vice versa)")
            hypothesis_validated_ar = True
        elif migrations_ar >= 5:
            print(f"  ✅ MODERATE: Peak conflict oscillated across {migrations_ar} layer transitions")
            hypothesis_validated_ar = True
        else:
            print(f"  ⚠️  WEAK: Limited peak migration detected ({migrations_ar} transitions)")

    if conflict_variance_ar:
        max_variance_ar = max(conflict_variance_ar.values())
        min_variance_ar = min(v for v in conflict_variance_ar.values() if v > 0) if any(v > 0 for v in conflict_variance_ar.values()) else 1
        if min_variance_ar > 0 and max_variance_ar / min_variance_ar > 5:
            print(f"  ✅ Conflict variance highly non-uniform (ratio: {max_variance_ar/min_variance_ar:.2f}x)")
            hypothesis_validated_ar = True

    # Combined Validation Summary
    print(f"\n{'='*80}")
    print("🏆 OVERALL HYPOTHESIS VALIDATION")
    print("="*80)

    if hypothesis_validated_bn and hypothesis_validated_ar:
        print("✅ STRONGLY VALIDATED: Both EN→BN and BN→AR show NON-STATIONARY conflicts")
        print("   → Static expert allocation is SUBOPTIMAL across sequential transfer")
        print("   → CADEA paper premise is SOUND for multilingual scenarios")
    elif hypothesis_validated_bn or hypothesis_validated_ar:
        print("✅ PARTIALLY VALIDATED: At least one stage shows NON-STATIONARY conflicts")
        validated_stage = "EN→BN" if hypothesis_validated_bn else "BN→AR"
        print(f"   → {validated_stage} demonstrates non-stationarity")
        print("   → CADEA may be beneficial but effect varies by language pair")
    else:
        print("❌ HYPOTHESIS WEAK: Both stages show relatively stationary conflicts")
        print("   → Consider longer training, different language pairs, or architecture")

    # ========================================================================
    # CATASTROPHIC FORGETTING ANALYSIS
    # ========================================================================
    print("\n" + "="*80)
    print("🔬 CATASTROPHIC FORGETTING ANALYSIS")
    print("="*80)

    # Only analyze if we have the required data
    has_en_data = 'after_en' in performance_tracking and 'en' in performance_tracking.get('after_en', {})
    has_bn_data = 'after_bn' in performance_tracking and 'bn' in performance_tracking.get('after_bn', {})
    has_ar_data = 'after_ar' in performance_tracking and 'ar' in performance_tracking.get('after_ar', {})
    
    if has_en_data and 'after_bn' in performance_tracking and 'en' in performance_tracking['after_bn']:
        en_initial_ppl = performance_tracking['after_en']['en']['perplexity']
        en_after_bn_ppl = performance_tracking['after_bn']['en']['perplexity']
        en_bn_degradation = ((en_after_bn_ppl - en_initial_ppl) / en_initial_ppl) * 100
        
        print(f"\n📉 ENGLISH FORGETTING (EN→BN):")
        print(f"  Initial (after EN): {en_initial_ppl:.2f}")
        print(f"  After BN: {en_after_bn_ppl:.2f} ({en_bn_degradation:+.1f}%)")
    
    if has_en_data and 'after_ar' in performance_tracking and 'en' in performance_tracking['after_ar']:
        en_initial_ppl = performance_tracking['after_en']['en']['perplexity']
        en_final_ppl = performance_tracking['after_ar']['en']['perplexity']
        en_total_degradation = ((en_final_ppl - en_initial_ppl) / en_initial_ppl) * 100
        
        if has_bn_data and 'en' in performance_tracking['after_bn']:
            print(f"  After AR: {en_final_ppl:.2f} ({en_total_degradation:+.1f}% total)")
        else:
            print(f"\n📉 ENGLISH FORGETTING (EN→AR):")
            print(f"  Initial (after EN): {en_initial_ppl:.2f}")
            print(f"  After AR: {en_final_ppl:.2f} ({en_total_degradation:+.1f}%)")

    if has_bn_data and 'after_ar' in performance_tracking and 'bn' in performance_tracking['after_ar']:
        bn_initial_ppl = performance_tracking['after_bn']['bn']['perplexity']
        bn_final_ppl = performance_tracking['after_ar']['bn']['perplexity']
        bn_degradation = ((bn_final_ppl - bn_initial_ppl) / bn_initial_ppl) * 100

        print(f"\n📉 BENGALI FORGETTING (BN→AR):")
        print(f"  Initial (after BN): {bn_initial_ppl:.2f}")
        print(f"  After AR: {bn_final_ppl:.2f} ({bn_degradation:+.1f}%)")

    # Calculate average forgetting if we have data
    degradations = []
    if has_en_data and 'after_ar' in performance_tracking and 'en' in performance_tracking['after_ar']:
        en_initial_ppl = performance_tracking['after_en']['en']['perplexity']
        en_final_ppl = performance_tracking['after_ar']['en']['perplexity']
        en_total_degradation = ((en_final_ppl - en_initial_ppl) / en_initial_ppl) * 100
        degradations.append(en_total_degradation)
    
    if has_bn_data and 'after_ar' in performance_tracking and 'bn' in performance_tracking['after_ar']:
        bn_initial_ppl = performance_tracking['after_bn']['bn']['perplexity']
        bn_final_ppl = performance_tracking['after_ar']['bn']['perplexity']
        bn_degradation = ((bn_final_ppl - bn_initial_ppl) / bn_initial_ppl) * 100
        degradations.append(bn_degradation)

    if degradations:
        avg_degradation = sum(degradations) / len(degradations)
        print(f"\n🎯 BACKWARD TRANSFER (BWT):")
        print(f"  Average forgetting: {avg_degradation:.1f}%")

        if avg_degradation > 10:
            print(f"\n✅ CATASTROPHIC FORGETTING CONFIRMED (>10% degradation)")
            print(f"   → This validates the need for dynamic expert allocation")
        else:
            print(f"\n⚠️  Mild forgetting observed (<10% degradation)")
            print(f"   → Static model may be more robust than expected")
    else:
        print(f"\n⚠️  Insufficient data for forgetting analysis (stages may have been skipped)")

    # ========================================================================
    # EXPORT RESULTS
    # ========================================================================
    print("\n" + "="*80)
    print("EXPORTING RESULTS")
    print("="*80)

    results = {
        'config': {
            'model': config.model_name,
            'languages': ['English', 'Bengali', 'Arabic'],
            'transfer_sequence': 'EN→BN→AR',
            'total_steps_per_stage': config.total_steps,
            'layers_tracked': config.layers_to_track,
            'learning_rate': config.learning_rate,
            'batch_size': config.batch_size,
            'gradient_accumulation': config.gradient_accumulation
        },

        'stage_1_en_bn': {
            'metrics': {
                'peak_migrations': int(migrations_bn),
                'unique_peak_layers': list(unique_peaks_bn),
                'initial_peak': int(peak_layers_bn[0]) if peak_layers_bn else None,
                'final_peak': int(peak_layers_bn[-1]) if peak_layers_bn else None,
                'layer_distance': int(layer_distance_bn) if peak_layers_bn else 0,
                'hypothesis_validated': bool(hypothesis_validated_bn)
            },
            'conflict_history': {
                str(layer): [
                    {k: float(v) if isinstance(v, (int, float, np.number)) else int(v)
                     for k, v in h.items()}
                    for h in history
                ]
                for layer, history in conflict_history_bn.items()
            },
            'training_metrics': {
                'steps': [int(s) for s in training_metrics_bn['steps']],
                'losses': [float(l) for l in training_metrics_bn['losses']],
                'peak_conflict_layers': [int(l) for l in training_metrics_bn['peak_conflict_layers']]
            },
            'conflict_variance': {str(k): float(v) for k, v in conflict_variance_bn.items()}
        },

        'stage_2_bn_ar': {
            'metrics': {
                'peak_migrations': int(migrations_ar),
                'unique_peak_layers': list(unique_peaks_ar),
                'initial_peak': int(peak_layers_ar[0]) if peak_layers_ar else None,
                'final_peak': int(peak_layers_ar[-1]) if peak_layers_ar else None,
                'layer_distance': int(layer_distance_ar) if peak_layers_ar else 0,
                'hypothesis_validated': bool(hypothesis_validated_ar)
            },
            'conflict_history': {
                str(layer): [
                    {k: float(v) if isinstance(v, (int, float, np.number)) else int(v)
                     for k, v in h.items()}
                    for h in history
                ]
                for layer, history in conflict_history_ar.items()
            },
            'training_metrics': {
                'steps': [int(s) for s in training_metrics_ar['steps']],
                'losses': [float(l) for l in training_metrics_ar['losses']],
                'peak_conflict_layers': [int(l) for l in training_metrics_ar['peak_conflict_layers']]
            },
            'conflict_variance': {str(k): float(v) for k, v in conflict_variance_ar.items()}
        },

        'overall_validation': {
            'both_stages_validated': bool(hypothesis_validated_bn and hypothesis_validated_ar),
            'at_least_one_validated': bool(hypothesis_validated_bn or hypothesis_validated_ar),
            'en_bn_validated': bool(hypothesis_validated_bn),
            'bn_ar_validated': bool(hypothesis_validated_ar)
        },

        'performance_tracking': {
            stage: {
                lang: {k: float(v) for k, v in metrics.items()}
                for lang, metrics in langs.items()
            }
            for stage, langs in performance_tracking.items()
        }
    }

    results_path = results_dir / 'cadea_sequential_transfer_data.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results exported to: {results_path}")

    performance_path = results_dir / 'performance_tracking.json'
    with open(performance_path, 'w') as f:
        tracking_serializable = {
            stage: {
                lang: {k: float(v) for k, v in metrics.items()}
                for lang, metrics in langs.items()
            }
            for stage, langs in performance_tracking.items()
        }
        json.dump(tracking_serializable, f, indent=2)
    print(f"✓ Performance tracking saved to: {performance_path}")

    vram_monitor.summary()
    print(f"\n🎉 Training complete! Checkpoints in: {config.checkpoint_dir}")
    print(f"   Results in: {config.results_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CADEA Local Training")
    parser.add_argument("--test", action="store_true", help="Run quick test (5 min)")
    parser.add_argument("--stage", choices=['en', 'bn', 'ar', 'full'], default='full',
                       help="Training stage to run")
    args = parser.parse_args()
    
    main(args)
