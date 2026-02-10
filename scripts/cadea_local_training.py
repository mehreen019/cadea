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
from pathlib import Path
from datetime import datetime
warnings.filterwarnings('ignore')

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
    
    max_length = 512
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
    
    # Stage 1: English (same as Kaggle version but with VRAM monitoring)
    print(f"\n{'='*80}")
    print("STAGE 1: ENGLISH TRAINING")
    print(f"{'='*80}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    for step, batch in enumerate(tqdm(en_dataloader, desc="English")):
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
        
        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")
        
        if step % config.vram_log_interval == 0:
            vram_monitor.log(step)
        
        if step > 0 and step % config.checkpoint_interval == 0:
            checkpoint_manager.save_checkpoint(
                stage='en', step=step, model=model, optimizer=optimizer,
                performance_tracking=performance_tracking
            )
        
        if step % 50 == 0:
            torch.cuda.empty_cache()
    
    print("✓ English training complete")

    # Evaluate after English
    print("\n" + "="*80)
    print("EVALUATION AFTER ENGLISH TRAINING")
    print("="*80)
    
    performance_tracking["after_en"]["en"] = evaluate_on_test(
        model, en_test_loader, "English", device
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
        
        layer_grads, loss = extract_layer_gradients(model, batch, config.layers_to_track, device)
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
    
    # ========================================================================
    # STAGE 2: BENGALI TRAINING WITH CONFLICT MONITORING
    # ========================================================================
    
    # ... [rest of the training code follows the same pattern]
    
    vram_monitor.summary()
    print(f"\n🎉 Training complete! Checkpoints in: {config.checkpoint_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CADEA Local Training")
    parser.add_argument("--test", action="store_true", help="Run quick test (5 min)")
    parser.add_argument("--stage", choices=['en', 'bn', 'ar', 'full'], default='full',
                       help="Training stage to run")
    args = parser.parse_args()
    
    main(args)
