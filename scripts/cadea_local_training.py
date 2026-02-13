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
    en_samples = 2000  # Increased from 100 for statistically meaningful baseline
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
    log_interval = 100  # Reduced frequency to help with memory
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
                       performance_tracking=None, baseline_grads=None, keep_only_latest=True):
        """Save complete training state and optionally clean up old checkpoints"""
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

        # Clean up old checkpoints for this stage (keep only latest)
        if keep_only_latest:
            self._cleanup_old_checkpoints(stage, checkpoint_path)

        return checkpoint_path

    def _cleanup_old_checkpoints(self, stage, current_checkpoint_path):
        """Delete old checkpoints for a stage, keeping only the current one"""
        # Find all checkpoints for this stage (excluding _complete checkpoints)
        pattern = f"{stage}_step_*.pt"
        old_checkpoints = list(self.checkpoint_dir.glob(pattern))

        deleted_count = 0
        freed_mb = 0

        for old_ckpt in old_checkpoints:
            if old_ckpt != current_checkpoint_path and old_ckpt.exists():
                size_mb = old_ckpt.stat().st_size / 1e6
                try:
                    old_ckpt.unlink()
                    deleted_count += 1
                    freed_mb += size_mb
                except OSError as e:
                    print(f"⚠️  Could not delete {old_ckpt.name}: {e}")

        if deleted_count > 0:
            print(f"🧹 Cleaned up {deleted_count} old checkpoint(s), freed {freed_mb:.1f} MB")
    
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
# EMA TRACKER FOR SMOOTHING
# ============================================================================
class EMATracker:
    """Exponential Moving Average tracker for smoothing noisy signals"""

    def __init__(self, beta=0.9):
        self.beta = beta
        self.value = None

    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.beta * self.value + (1 - self.beta) * new_value
        return self.value

    def reset(self):
        self.value = None

# ============================================================================
# CONFLICT MONITOR WITH NOISE FLOOR CALIBRATION
# ============================================================================
class ConflictMonitor:
    """
    Monitor gradient conflicts with noise floor subtraction and magnitude weighting.

    Key features:
    1. Calibrates noise floor using EN-EN conflict baseline
    2. Computes Normalized Conflict Score (NCS) by subtracting noise floor
    3. Computes allocation signal weighted by gradient magnitude
    4. Uses EMA smoothing for stable peak detection
    """

    def __init__(self, layers_to_track, default_noise_floor=0.8, ema_beta=0.9, allocation_ema_beta=0.95):
        self.layers_to_track = layers_to_track
        self.default_noise_floor = default_noise_floor

        # Noise floor per layer (calibrated from EN-EN)
        self.noise_floor = {layer: default_noise_floor for layer in layers_to_track}
        self.noise_calibrated = False

        # EMA trackers for each layer
        self.ema_conflict = {layer: EMATracker(beta=ema_beta) for layer in layers_to_track}
        self.ema_en_norm = {layer: EMATracker(beta=ema_beta) for layer in layers_to_track}
        self.ema_new_lang_norm = {layer: EMATracker(beta=ema_beta) for layer in layers_to_track}
        self.ema_allocation = {layer: EMATracker(beta=allocation_ema_beta) for layer in layers_to_track}
        self.ema_ncs = {layer: EMATracker(beta=ema_beta) for layer in layers_to_track}

        # History for diagnostics
        self.diagnostic_history = []

    def calibrate_noise_floor_from_en_en_conflicts(self, en_en_conflicts):
        """
        Calibrate noise floor from pre-computed EN-EN conflicts.

        Args:
            en_en_conflicts: Dict mapping layer_idx -> EN-EN conflict score
        """
        for layer, conflict in en_en_conflicts.items():
            if layer in self.layers_to_track:
                self.noise_floor[layer] = conflict

        self.noise_calibrated = True

        # Validation
        avg_noise = np.mean(list(self.noise_floor.values()))
        print(f"✓ Noise floor calibrated from EN-EN conflicts")
        print(f"  Average noise floor: {avg_noise:.4f}")
        for layer in sorted(self.noise_floor.keys()):
            print(f"  Layer {layer}: {self.noise_floor[layer]:.4f}")

    def compute_normalized_conflict(self, raw_conflict, layer_idx):
        """
        Subtract noise floor to get Normalized Conflict Score (NCS).

        NCS = (raw_conflict - baseline) / (2.0 - baseline)

        Returns value in [0, 1] range representing TRUE excess conflict.
        """
        if not self.noise_calibrated:
            # Fallback: use default noise floor
            baseline = self.default_noise_floor
        else:
            baseline = self.noise_floor.get(layer_idx, self.default_noise_floor)

        # Normalized Conflict Score
        # Scale to [0, 1] where 0 = no excess conflict, 1 = maximum conflict
        ncs = (raw_conflict - baseline) / (2.0 - baseline)
        return max(0.0, ncs)  # Only positive excess conflict

    def compute_allocation_signal(self, layer_idx, en_norm, new_lang_norm, normalized_conflict, lambda_weight=0.5):
        """
        Compute weighted allocation signal.

        S_l = ||∇L_new||_l × (1 + λ × NCS_l)

        Args:
            layer_idx: Layer index
            en_norm: English gradient norm for this layer
            new_lang_norm: New language gradient norm for this layer
            normalized_conflict: NCS value (noise-floor subtracted)
            lambda_weight: Conflict modulation strength (0.5 recommended)

        Returns:
            Allocation signal value
        """
        # Primary signal: magnitude of new language gradient (necessity)
        base_signal = new_lang_norm

        # Modulation: boost if conflict exists (interference)
        conflict_multiplier = 1 + lambda_weight * normalized_conflict

        allocation_signal = base_signal * conflict_multiplier

        return allocation_signal

    def update_ema_trackers(self, layer_idx, raw_conflict, ncs, en_norm, new_lang_norm, allocation_signal):
        """Update all EMA trackers for a layer"""
        self.ema_conflict[layer_idx].update(raw_conflict)
        self.ema_ncs[layer_idx].update(ncs)
        self.ema_en_norm[layer_idx].update(en_norm)
        self.ema_new_lang_norm[layer_idx].update(new_lang_norm)
        self.ema_allocation[layer_idx].update(allocation_signal)

    def get_smoothed_allocation_signals(self):
        """Get current smoothed allocation signals for all layers"""
        return {
            layer: self.ema_allocation[layer].value
            for layer in self.layers_to_track
            if self.ema_allocation[layer].value is not None
        }

    def select_peak_layer_robust(self, allocation_signals=None, min_threshold=0.1):
        """
        Select peak layer using smoothed, weighted signals.

        Only considers layers above threshold to avoid noise.

        Args:
            allocation_signals: Dict of layer -> signal (uses smoothed if None)
            min_threshold: Minimum signal to consider a layer

        Returns:
            Peak layer index or None if no clear signal
        """
        if allocation_signals is None:
            allocation_signals = self.get_smoothed_allocation_signals()

        if not allocation_signals:
            return None

        # Filter to valid layers above threshold
        valid_layers = {
            layer: signal
            for layer, signal in allocation_signals.items()
            if signal is not None and signal > min_threshold
        }

        if not valid_layers:
            return None

        return max(valid_layers, key=valid_layers.get)

    def log_diagnostic(self, step, layer_idx, raw_conflict, ncs, en_norm, new_lang_norm,
                       allocation_signal, lambda_weight=0.5):
        """Log comprehensive diagnostic data for a layer"""
        diagnostic = {
            'step': step,
            'layer_idx': layer_idx,

            # Raw signals
            'raw_conflict': float(raw_conflict),
            'raw_en_norm': float(en_norm),
            'raw_new_lang_norm': float(new_lang_norm),

            # Processed signals
            'noise_floor': float(self.noise_floor.get(layer_idx, self.default_noise_floor)),
            'normalized_conflict_score': float(ncs),
            'allocation_signal': float(allocation_signal),

            # Smoothed signals
            'ema_conflict': float(self.ema_conflict[layer_idx].value) if self.ema_conflict[layer_idx].value else None,
            'ema_ncs': float(self.ema_ncs[layer_idx].value) if self.ema_ncs[layer_idx].value else None,
            'ema_en_norm': float(self.ema_en_norm[layer_idx].value) if self.ema_en_norm[layer_idx].value else None,
            'ema_new_lang_norm': float(self.ema_new_lang_norm[layer_idx].value) if self.ema_new_lang_norm[layer_idx].value else None,
            'ema_allocation': float(self.ema_allocation[layer_idx].value) if self.ema_allocation[layer_idx].value else None,

            # Ratios for analysis
            'norm_ratio_new_to_en': float(new_lang_norm / (en_norm + 1e-8)),
            'conflict_contribution': float(lambda_weight * ncs),
        }

        self.diagnostic_history.append(diagnostic)
        return diagnostic

    def get_validation_checks(self):
        """Return validation check results"""
        checks = {}

        # Check noise floor is in expected range
        noise_values = list(self.noise_floor.values())
        checks['noise_floor_valid'] = all(0.5 <= nf <= 0.95 for nf in noise_values)
        checks['noise_floor_avg'] = np.mean(noise_values)

        # Check for smoothed signal availability
        allocation_values = [v.value for v in self.ema_allocation.values() if v.value is not None]
        if allocation_values:
            checks['allocation_range'] = (min(allocation_values), max(allocation_values))
            checks['allocation_ratio'] = max(allocation_values) / (min(allocation_values) + 1e-8)

        return checks

    def reset_for_new_stage(self):
        """Reset EMA trackers for a new training stage"""
        for layer in self.layers_to_track:
            self.ema_conflict[layer].reset()
            self.ema_ncs[layer].reset()
            self.ema_en_norm[layer].reset()
            self.ema_new_lang_norm[layer].reset()
            self.ema_allocation[layer].reset()

        self.diagnostic_history = []

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
        # Track BOTH MLP and Attention layers (Fix #3: was missing ~40% of parameters)
        if "model.layers." in name and (".mlp." in name or ".self_attn." in name):
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


def compute_current_task_gradient(model, dataloader, layers_to_track, device, tokenizer, num_batches=2):
    """
    Compute gradient for a task at CURRENT model state WITHOUT updating model.

    This is the FIX for the static baseline problem - we compute what the comparison
    task's gradient would be at the current model state, allowing us to measure
    TRUE interference rather than comparing to a frozen snapshot.

    MEMORY OPTIMIZED for 8GB VRAM:
    - Uses only 2 batches (reduced from 5)
    - Aggressive cache clearing between batches
    - Handles OOM gracefully by returning None (caller should use static fallback)

    Args:
        model: The model to compute gradients for
        dataloader: DataLoader for the comparison task
        layers_to_track: Which layers to extract gradients from
        device: Torch device
        tokenizer: Tokenizer for the model
        num_batches: Number of batches to average over (default 2 for memory)

    Returns:
        Dict mapping layer numbers to averaged gradient tensors (on GPU),
        or None if OOM occurred (caller should fall back to static baseline for this step)
    """
    was_training = model.training

    # Clear memory before starting
    model.zero_grad()
    torch.cuda.empty_cache()

    try:
        model.eval()

        task_grads = {layer: [] for layer in layers_to_track}
        batch_iter = iter(dataloader)

        for i in range(min(num_batches, len(dataloader))):
            try:
                batch = next(batch_iter)
            except StopIteration:
                batch_iter = iter(dataloader)
                batch = next(batch_iter)

            layer_grads, _ = extract_layer_gradients(model, batch, layers_to_track, device, tokenizer)

            if layer_grads is not None:
                for layer, grad in layer_grads.items():
                    if grad is not None and grad.numel() > 1:
                        task_grads[layer].append(grad.clone())

            # Aggressive cleanup after each batch
            model.zero_grad()
            del batch, layer_grads
            torch.cuda.empty_cache()

        # Average across batches
        averaged_grads = {}
        for layer, grads_list in task_grads.items():
            if grads_list:
                averaged_grads[layer] = torch.mean(torch.stack(grads_list), dim=0)
            else:
                averaged_grads[layer] = torch.zeros(1, device=device)

        return averaged_grads

    except torch.cuda.OutOfMemoryError:
        # OOM - signal caller to use static baseline for THIS checkpoint only
        print("\n⚠️  OOM in dynamic baseline, using static baseline for this step")
        model.zero_grad()
        torch.cuda.empty_cache()
        gc.collect()
        return None

    finally:
        if was_training:
            model.train()
        model.zero_grad()
        torch.cuda.empty_cache()


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================
def main(args):
    # Set seed for reproducibility
    seed = args.seed
    set_seed(seed)
    print(f"🌱 Random seed set to: {seed}")

    # Select config
    if args.test:
        config = TestConfig()
        print("🧪 Using TEST configuration (5-minute run)")
    else:
        config = Config()
        print("🚀 Using FULL configuration")
    
    # Override config with command-line arguments
    if args.en_samples is not None:
        config.en_samples = args.en_samples
        print(f"📝 Overriding EN samples: {config.en_samples}")
    if args.bn_samples is not None:
        config.bn_samples = args.bn_samples
        print(f"📝 Overriding BN samples: {config.bn_samples}")
    if args.ar_samples is not None:
        config.ar_samples = args.ar_samples
        print(f"📝 Overriding AR samples: {config.ar_samples}")
    if args.batch_size is not None:
        config.batch_size = args.batch_size
        print(f"📝 Overriding batch size: {config.batch_size}")
    
    # Checkpointing control
    enable_checkpointing = args.checkpointing == 'yes'
    if not enable_checkpointing:
        print("⚠️  Checkpointing DISABLED (running in no-checkpoint mode)")
    
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

    # Training subset loaders for TRUE forgetting measurement (Fix #6)
    # Catastrophic forgetting = degradation on TRAINING data, not just test data
    en_train_eval_size = min(200, len(en_dataset))
    bn_train_eval_size = min(200, len(bn_dataset))
    ar_train_eval_size = min(200, len(ar_dataset))

    # Use Subset to get first N samples from training sets
    from torch.utils.data import Subset
    en_train_eval = Subset(en_dataset, range(en_train_eval_size))
    bn_train_eval = Subset(bn_dataset, range(bn_train_eval_size))
    ar_train_eval = Subset(ar_dataset, range(ar_train_eval_size))

    en_train_eval_loader = DataLoader(en_train_eval, batch_size=config.batch_size, shuffle=False)
    bn_train_eval_loader = DataLoader(bn_train_eval, batch_size=config.batch_size, shuffle=False)
    ar_train_eval_loader = DataLoader(ar_train_eval, batch_size=config.batch_size, shuffle=False)
    
    # Initialize tracking
    performance_tracking = {"after_en": {}, "after_bn": {}, "after_ar": {}}
    
    # Check for resume
    if enable_checkpointing:
        checkpoint_manager.list_checkpoints()

    # Determine resume point
    resume_stage = None
    resume_step = 0
    
    if enable_checkpointing:
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
    else:
        print("⏭️  Skipping checkpoint detection (checkpointing disabled)")

    # Stage 1: English (same as Kaggle version but with VRAM monitoring)
    skip_en = resume_stage in ['en_complete', 'bn', 'bn_complete', 'ar', 'ar_complete']
    if args.stage in ['bn', 'ar']:
        skip_en = True  # User requested to skip English

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

                if enable_checkpointing and step > 0 and step % config.checkpoint_interval == 0:
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

        # Test set evaluation
        performance_tracking["after_en"]["en_test"] = evaluate_on_test(
            model, en_test_loader, "English Test", device, tokenizer
        )
        # Training set evaluation (for true forgetting measurement)
        performance_tracking["after_en"]["en_train"] = evaluate_on_test(
            model, en_train_eval_loader, "English Train", device, tokenizer
        )
        print(f"EN Test  - Loss: {performance_tracking['after_en']['en_test']['loss']:.4f}, "
              f"Perplexity: {performance_tracking['after_en']['en_test']['perplexity']:.2f}")
        print(f"EN Train - Loss: {performance_tracking['after_en']['en_train']['loss']:.4f}, "
              f"Perplexity: {performance_tracking['after_en']['en_train']['perplexity']:.2f}")

        # Save post-English checkpoint
        if enable_checkpointing:
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
        else:
            print("⏭️  Skipping checkpoint save (checkpointing disabled)")

        # Compute English baseline gradients
        print("\n" + "="*80)
        print("COMPUTING ENGLISH BASELINE GRADIENT PROFILE")
        print("="*80)

        english_baseline_grads = {layer: None for layer in config.layers_to_track}
        en_losses = []

        num_baseline_batches = min(100, len(en_dataloader))  # Increased for stable baseline

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
        if enable_checkpointing:
            baseline_path = checkpoint_manager.checkpoint_dir / "english_baseline_grads.pt"
            torch.save(english_baseline_grads, baseline_path)
            print(f"✓ Baseline gradients saved: {baseline_path}")
        else:
            print("⏭️  Skipping baseline gradient save (checkpointing disabled)")
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
            num_baseline_batches = min(100, len(en_dataloader))  # Increased for stable baseline
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
    # EN-EN SANITY CHECK: Compute conflict between two English batch groups
    # ========================================================================
    # This validates whether cross-language conflict is real signal or noise
    # Expected: EN-EN conflict < 0.3 → EN-BN 0.8+ is real signal ✅
    #           EN-EN conflict > 0.6 → Gradients are naturally noisy ❌

    print("\n" + "="*80)
    print("EN-EN SANITY CHECK: Validating Gradient Conflict Baseline")
    print("="*80)

    # Split English data into two groups
    en_group_a_grads = {layer: None for layer in config.layers_to_track}
    en_group_b_grads = {layer: None for layer in config.layers_to_track}

    # Use even batches for group A, odd batches for group B
    num_sanity_batches = min(40, len(en_dataloader))  # 20 per group
    group_a_count = 0
    group_b_count = 0

    print(f"Computing gradients from {num_sanity_batches} English batches (split into 2 groups)...")

    for i, batch in enumerate(tqdm(en_dataloader, total=num_sanity_batches, desc="EN-EN sanity check")):
        if i >= num_sanity_batches:
            break

        layer_grads, _ = extract_layer_gradients(model, batch, config.layers_to_track, device, tokenizer)
        if layer_grads is None:
            continue

        # Alternate between groups
        if i % 2 == 0:
            for layer, grad in layer_grads.items():
                if en_group_a_grads[layer] is None:
                    en_group_a_grads[layer] = grad.clone()
                else:
                    en_group_a_grads[layer] += grad
            group_a_count += 1
        else:
            for layer, grad in layer_grads.items():
                if en_group_b_grads[layer] is None:
                    en_group_b_grads[layer] = grad.clone()
                else:
                    en_group_b_grads[layer] += grad
            group_b_count += 1

        model.zero_grad()

        if i % 10 == 0:
            torch.cuda.empty_cache()

    # Average each group
    for layer in config.layers_to_track:
        if en_group_a_grads[layer] is not None and group_a_count > 0:
            en_group_a_grads[layer] /= group_a_count
        if en_group_b_grads[layer] is not None and group_b_count > 0:
            en_group_b_grads[layer] /= group_b_count

    # Compute EN-EN conflict (cosine similarity between two English groups)
    print(f"\n📊 EN-EN CONFLICT SCORES (lower = less noisy gradients):")
    print("-" * 60)

    en_en_conflicts = {}
    for layer in config.layers_to_track:
        if (en_group_a_grads[layer] is not None and
            en_group_b_grads[layer] is not None and
            en_group_a_grads[layer].numel() > 1 and
            en_group_b_grads[layer].numel() > 1):

            # Normalize and compute cosine similarity
            grad_a_norm = F.normalize(en_group_a_grads[layer].unsqueeze(0), dim=1)
            grad_b_norm = F.normalize(en_group_b_grads[layer].unsqueeze(0), dim=1)
            cos_sim = F.cosine_similarity(grad_a_norm, grad_b_norm).item()
            conflict_score = 1 - cos_sim
            en_en_conflicts[layer] = conflict_score

            print(f"  Layer {layer:2d}: conflict = {conflict_score:.4f}  (cos_sim = {cos_sim:.4f})")

    # Summary statistics
    if en_en_conflicts:
        avg_en_en_conflict = np.mean(list(en_en_conflicts.values()))
        max_en_en_conflict = max(en_en_conflicts.values())
        min_en_en_conflict = min(en_en_conflicts.values())

        print("-" * 60)
        print(f"  Average EN-EN conflict: {avg_en_en_conflict:.4f}")
        print(f"  Range: [{min_en_en_conflict:.4f}, {max_en_en_conflict:.4f}]")

        # Interpretation
        print("\n🔬 INTERPRETATION:")
        if avg_en_en_conflict < 0.3:
            print(f"  ✅ EN-EN conflict is LOW ({avg_en_en_conflict:.3f} < 0.3)")
            print(f"     → Same-language gradients are consistent")
            print(f"     → If EN-BN conflict > 0.8, it's REAL cross-lingual interference!")
        elif avg_en_en_conflict < 0.6:
            print(f"  ⚠️  EN-EN conflict is MODERATE ({avg_en_en_conflict:.3f})")
            print(f"     → Some natural variance in gradients")
            print(f"     → EN-BN conflict needs to be significantly higher to be meaningful")
        else:
            print(f"  ❌ EN-EN conflict is HIGH ({avg_en_en_conflict:.3f} > 0.6)")
            print(f"     → Gradients are naturally noisy")
            print(f"     → Cross-language conflict may not be distinguishable from noise")
            print(f"     → Consider: larger batches, more accumulation steps, or gradient smoothing")

    # Clean up sanity check tensors
    del en_group_a_grads, en_group_b_grads
    torch.cuda.empty_cache()

    print("=" * 80 + "\n")

    # ========================================================================
    # INITIALIZE CONFLICT MONITOR WITH NOISE FLOOR CALIBRATION
    # ========================================================================
    print("=" * 80)
    print("INITIALIZING CONFLICT MONITOR WITH NOISE FLOOR CALIBRATION")
    print("=" * 80)

    conflict_monitor = ConflictMonitor(
        layers_to_track=config.layers_to_track,
        default_noise_floor=0.8,
        ema_beta=0.9,
        allocation_ema_beta=0.95
    )

    # Calibrate noise floor from EN-EN conflicts computed above
    if en_en_conflicts:
        conflict_monitor.calibrate_noise_floor_from_en_en_conflicts(en_en_conflicts)

        # Validation checks
        validation = conflict_monitor.get_validation_checks()
        if validation['noise_floor_valid']:
            print(f"✓ Noise floor validation PASSED (avg: {validation['noise_floor_avg']:.4f})")
        else:
            print(f"⚠️  Noise floor validation WARNING (avg: {validation['noise_floor_avg']:.4f})")
            print(f"   Expected range: [0.5, 0.95] - consider more EN-EN batches")
    else:
        print("⚠️  No EN-EN conflicts computed, using default noise floor (0.8)")

    print("=" * 80 + "\n")

    # ========================================================================
    # STAGE 2: BENGALI TRAINING WITH CONFLICT MONITORING
    # ========================================================================
    skip_bn = resume_stage in ['bn_complete', 'ar', 'ar_complete']
    if args.stage == 'en':
        skip_bn = True  # User only wants English
    if args.stage == 'ar':
        skip_bn = True  # User wants to skip to Arabic

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
        'peak_conflict_layers': [],
        'skipped_steps_oom': []  # Track steps skipped due to OOM
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

            # === CONFLICT MONITORING (DYNAMIC BASELINE - computes EN grad at CURRENT model state) ===
            if step % config.log_interval == 0:
                # Extract Bengali gradients directly from current backward pass
                layer_grads = {}
                for name, param in model.named_parameters():
                    if param.grad is None:
                        continue

                    # Track BOTH MLP and Attention layers
                    if ".layers." in name and (".mlp." in name or ".self_attn." in name):
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

                # MEMORY FIX: Clear Bengali forward pass memory BEFORE computing English grads
                del input_ids, attention_mask, labels, outputs, loss
                model.zero_grad()
                torch.cuda.empty_cache()

                # CRITICAL FIX: Compute CURRENT English gradient at this model state
                # This measures TRUE interference, not comparison to frozen snapshot
                current_en_grads = compute_current_task_gradient(
                    model, en_dataloader, config.layers_to_track, device, tokenizer, num_batches=2
                )

                # If OOM occurred, skip conflict logging for this step (don't use bad data)
                if current_en_grads is None:
                    training_metrics_bn['skipped_steps_oom'].append(step)
                    accumulated_loss = 0
                    continue

                # Compute conflict with NOISE FLOOR SUBTRACTION and MAGNITUDE WEIGHTING
                layer_conflicts = {}
                layer_ncs = {}  # Normalized Conflict Scores
                layer_allocation_signals = {}
                lambda_weight = 0.5  # Conflict modulation strength

                for layer in config.layers_to_track:
                    if layer in bn_layer_grads and layer in current_en_grads:
                        en_grad = current_en_grads[layer]
                        bn_grad = bn_layer_grads[layer]

                        # Skip if either gradient is a placeholder
                        if en_grad.numel() <= 1 or bn_grad.numel() <= 1:
                            continue

                        # Compute gradient norms (magnitude)
                        en_norm = en_grad.norm().item()
                        bn_norm = bn_grad.norm().item()

                        # Normalize gradients for angle computation
                        en_grad_normalized = F.normalize(en_grad.unsqueeze(0), dim=1)
                        bn_grad_normalized = F.normalize(bn_grad.unsqueeze(0), dim=1)

                        # Cosine similarity and raw conflict
                        cos_sim = F.cosine_similarity(en_grad_normalized, bn_grad_normalized).item()
                        raw_conflict = 1 - cos_sim

                        # Compute Normalized Conflict Score (subtract noise floor)
                        ncs = conflict_monitor.compute_normalized_conflict(raw_conflict, layer)
                        layer_ncs[layer] = ncs

                        # Compute allocation signal (magnitude-weighted)
                        allocation_signal = conflict_monitor.compute_allocation_signal(
                            layer_idx=layer,
                            en_norm=en_norm,
                            new_lang_norm=bn_norm,
                            normalized_conflict=ncs,
                            lambda_weight=lambda_weight
                        )
                        layer_allocation_signals[layer] = allocation_signal

                        # Update EMA trackers
                        conflict_monitor.update_ema_trackers(
                            layer_idx=layer,
                            raw_conflict=raw_conflict,
                            ncs=ncs,
                            en_norm=en_norm,
                            new_lang_norm=bn_norm,
                            allocation_signal=allocation_signal
                        )

                        # Log diagnostic data
                        conflict_monitor.log_diagnostic(
                            step=step,
                            layer_idx=layer,
                            raw_conflict=raw_conflict,
                            ncs=ncs,
                            en_norm=en_norm,
                            new_lang_norm=bn_norm,
                            allocation_signal=allocation_signal,
                            lambda_weight=lambda_weight
                        )

                        # Store comprehensive conflict metrics
                        conflict_history_bn[layer].append({
                            'step': step,
                            'cosine_similarity': cos_sim,
                            'conflict_score': raw_conflict,
                            'normalized_conflict_score': ncs,
                            'allocation_signal': allocation_signal,
                            'bn_grad_norm': bn_norm,
                            'en_grad_norm': en_norm,
                            'noise_floor': conflict_monitor.noise_floor.get(layer, 0.8),
                            'ema_ncs': conflict_monitor.ema_ncs[layer].value,
                            'ema_allocation': conflict_monitor.ema_allocation[layer].value,
                            'norm_ratio_bn_to_en': bn_norm / (en_norm + 1e-8),
                        })

                        layer_conflicts[layer] = raw_conflict

                # Find peak layer using SMOOTHED ALLOCATION SIGNALS (not raw conflict)
                if layer_allocation_signals:
                    # Use smoothed allocation signals for robust peak selection
                    smoothed_signals = conflict_monitor.get_smoothed_allocation_signals()

                    # For early steps when EMA isn't populated, use raw signals
                    if not smoothed_signals or all(v is None for v in smoothed_signals.values()):
                        smoothed_signals = layer_allocation_signals

                    peak_layer = conflict_monitor.select_peak_layer_robust(
                        allocation_signals=smoothed_signals,
                        min_threshold=0.01  # Lower threshold for early training
                    )

                    # Fallback to raw max if no valid peak
                    if peak_layer is None and layer_allocation_signals:
                        peak_layer = max(layer_allocation_signals, key=layer_allocation_signals.get)

                    if peak_layer is not None:
                        peak_allocation = smoothed_signals.get(peak_layer, layer_allocation_signals.get(peak_layer, 0))
                        max_ncs = max(layer_ncs.values()) if layer_ncs else 0

                        # Compute confidence as ratio of peak to second highest
                        sorted_signals = sorted(smoothed_signals.items(), key=lambda x: x[1] if x[1] else 0, reverse=True)
                        if len(sorted_signals) > 1 and sorted_signals[1][1]:
                            peak_confidence = sorted_signals[0][1] / (sorted_signals[1][1] + 1e-8)
                        else:
                            peak_confidence = 1.0

                        training_metrics_bn['peak_conflict_layers'].append(peak_layer)
                        training_metrics_bn['steps'].append(step)
                        training_metrics_bn['losses'].append(accumulated_loss * config.gradient_accumulation)

                        # Store additional metrics
                        if 'peak_confidence' not in training_metrics_bn:
                            training_metrics_bn['peak_confidence'] = []
                        training_metrics_bn['peak_confidence'].append(peak_confidence)

                        if 'peak_allocation_signal' not in training_metrics_bn:
                            training_metrics_bn['peak_allocation_signal'] = []
                        training_metrics_bn['peak_allocation_signal'].append(peak_allocation)

                        if 'max_ncs' not in training_metrics_bn:
                            training_metrics_bn['max_ncs'] = []
                        training_metrics_bn['max_ncs'].append(max_ncs)

                        accumulated_loss = 0

                        # Log with new metrics
                        progress_bar.set_postfix({
                            'loss': f"{training_metrics_bn['losses'][-1]:.4f}",
                            'peak_L': peak_layer,
                            'max_NCS': f"{max_ncs:.3f}",
                            'alloc': f"{peak_allocation:.2e}",
                            'conf': f"{peak_confidence:.2f}"
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
            if enable_checkpointing and step > 0 and step % config.checkpoint_interval == 0:
                checkpoint_manager.save_checkpoint(
                    stage='bn', step=step, model=model, optimizer=optimizer,
                    scheduler=scheduler, conflict_history=conflict_history_bn,
                    training_metrics=training_metrics_bn,
                    performance_tracking=performance_tracking,
                    baseline_grads=english_baseline_grads
                )

        progress_bar.close()
        print("\n✓ Bengali training complete")

        # Report any skipped steps
        if training_metrics_bn['skipped_steps_oom']:
            print(f"⚠️  {len(training_metrics_bn['skipped_steps_oom'])} conflict logging steps skipped due to OOM")
            print(f"   Skipped at steps: {training_metrics_bn['skipped_steps_oom']}")

        # Evaluate after Bengali
        print("\n" + "="*80)
        print("EVALUATION AFTER BENGALI TRAINING")
        print("="*80)

        # Test set evaluations
        performance_tracking["after_bn"]["en_test"] = evaluate_on_test(
            model, en_test_loader, "English Test", device, tokenizer
        )
        performance_tracking["after_bn"]["bn_test"] = evaluate_on_test(
            model, bn_test_loader, "Bengali Test", device, tokenizer
        )
        # Training set evaluations (for true forgetting measurement)
        performance_tracking["after_bn"]["en_train"] = evaluate_on_test(
            model, en_train_eval_loader, "English Train", device, tokenizer
        )
        performance_tracking["after_bn"]["bn_train"] = evaluate_on_test(
            model, bn_train_eval_loader, "Bengali Train", device, tokenizer
        )

        print(f"EN Test  - Loss: {performance_tracking['after_bn']['en_test']['loss']:.4f}, "
              f"Perplexity: {performance_tracking['after_bn']['en_test']['perplexity']:.2f}")
        print(f"EN Train - Loss: {performance_tracking['after_bn']['en_train']['loss']:.4f}, "
              f"Perplexity: {performance_tracking['after_bn']['en_train']['perplexity']:.2f}")
        print(f"BN Test  - Loss: {performance_tracking['after_bn']['bn_test']['loss']:.4f}, "
              f"Perplexity: {performance_tracking['after_bn']['bn_test']['perplexity']:.2f}")
        print(f"BN Train - Loss: {performance_tracking['after_bn']['bn_train']['loss']:.4f}, "
              f"Perplexity: {performance_tracking['after_bn']['bn_train']['perplexity']:.2f}")

        # Calculate forgetting on TRAINING set (true catastrophic forgetting)
        en_train_degradation = ((performance_tracking['after_bn']['en_train']['perplexity'] -
                                  performance_tracking['after_en']['en_train']['perplexity']) /
                                 performance_tracking['after_en']['en_train']['perplexity']) * 100
        en_test_degradation = ((performance_tracking['after_bn']['en_test']['perplexity'] -
                                 performance_tracking['after_en']['en_test']['perplexity']) /
                                performance_tracking['after_en']['en_test']['perplexity']) * 100
        print(f"\n⚠️  EN Train Forgetting: {en_train_degradation:+.1f}% (TRUE catastrophic forgetting)")
        print(f"⚠️  EN Test Degradation: {en_test_degradation:+.1f}% (generalization loss)")

        # Save post-Bengali checkpoint
        if enable_checkpointing:
            checkpoint_manager.save_checkpoint(
                stage='bn_complete', step=config.total_steps, model=model, optimizer=optimizer,
                scheduler=scheduler, conflict_history=conflict_history_bn,
                training_metrics=training_metrics_bn,
                performance_tracking=performance_tracking,
                baseline_grads=english_baseline_grads
            )
        else:
            print("⏭️  Skipping checkpoint save (checkpointing disabled)")

        # ========================================================================
        # COMPUTE BENGALI BASELINE GRADIENT PROFILE
        # ========================================================================
        print("\n" + "="*80)
        print("COMPUTING BENGALI BASELINE GRADIENT PROFILE (post-training)")
        print("="*80)

        bengali_baseline_grads = {layer: None for layer in config.layers_to_track}
        bn_baseline_losses = []

        num_baseline_batches = min(100, len(bn_dataloader))  # Increased for stable baseline

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
        if enable_checkpointing:
            bn_baseline_path = checkpoint_manager.checkpoint_dir / "bengali_baseline_grads.pt"
            torch.save(bengali_baseline_grads, bn_baseline_path)
            print(f"✓ Bengali baseline gradients saved: {bn_baseline_path}")
        else:
            print("⏭️  Skipping baseline gradient save (checkpointing disabled)")
    else:
        # Load Bengali baseline gradients from file
        bn_baseline_path = checkpoint_manager.checkpoint_dir / "bengali_baseline_grads.pt"
        if bn_baseline_path.exists():
            bengali_baseline_grads = torch.load(bn_baseline_path, map_location=device)
            print(f"✓ Loaded Bengali baseline gradients from: {bn_baseline_path}")
        else:
            print("⚠️  Bengali baseline gradients not found, will recompute...")
            bengali_baseline_grads = {layer: None for layer in config.layers_to_track}
            num_baseline_batches = min(100, len(bn_dataloader))  # Increased for stable baseline
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
    skip_ar = args.stage in ['en', 'bn']
    if skip_ar:
        print(f"\n{'='*80}")
        print("STAGE 3: ARABIC TRAINING - SKIPPED (--stage flag)")
        print(f"{'='*80}")
    else:
        print(f"\n{'='*80}")
        print("STAGE 3: ARABIC TRAINING WITH CONFLICT MONITORING (vs Bengali baseline)")
        print(f"{'='*80}")

    # Tracking structures (needed even when skipping for export)
    conflict_history_ar = {layer: [] for layer in config.layers_to_track}
    training_metrics_ar = {
        'steps': [],
        'losses': [],
        'peak_conflict_layers': [],
        'skipped_steps_oom': []  # Track steps skipped due to OOM
    }

    if not skip_ar:
        # Reset ConflictMonitor EMA trackers for new stage
        conflict_monitor.reset_for_new_stage()
        print("✓ ConflictMonitor reset for Arabic stage (EMA trackers cleared)")

        # RE-INITIALIZE optimizer (fresh training phase)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=50,
            num_training_steps=config.total_steps
        )

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

            # === CONFLICT MONITORING (DYNAMIC BASELINE - computes BN grad at CURRENT model state) ===
            if step % config.log_interval == 0:
                # Extract Arabic gradients directly from current backward pass
                layer_grads = {}
                for name, param in model.named_parameters():
                    if param.grad is None:
                        continue

                    # Track BOTH MLP and Attention layers
                    if ".layers." in name and (".mlp." in name or ".self_attn." in name):
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

                # MEMORY FIX: Clear Arabic forward pass memory BEFORE computing Bengali grads
                del input_ids, attention_mask, labels, outputs, loss
                model.zero_grad()
                torch.cuda.empty_cache()

                # CRITICAL FIX: Compute CURRENT Bengali gradient at this model state
                # This measures TRUE interference, not comparison to frozen snapshot
                current_bn_grads = compute_current_task_gradient(
                    model, bn_dataloader, config.layers_to_track, device, tokenizer, num_batches=2
                )

                # If OOM occurred, skip conflict logging for this step (don't use bad data)
                if current_bn_grads is None:
                    training_metrics_ar['skipped_steps_oom'].append(step)
                    accumulated_loss = 0
                    continue

                # Compute conflict with NOISE FLOOR SUBTRACTION and MAGNITUDE WEIGHTING
                layer_conflicts = {}
                layer_ncs = {}  # Normalized Conflict Scores
                layer_allocation_signals = {}
                lambda_weight = 0.5  # Conflict modulation strength

                for layer in config.layers_to_track:
                    if layer in ar_layer_grads and layer in current_bn_grads:
                        bn_grad = current_bn_grads[layer]
                        ar_grad = ar_layer_grads[layer]

                        # Skip if either gradient is a placeholder
                        if bn_grad.numel() <= 1 or ar_grad.numel() <= 1:
                            continue

                        # Compute gradient norms (magnitude)
                        bn_norm = bn_grad.norm().item()
                        ar_norm = ar_grad.norm().item()

                        # Normalize gradients for angle computation
                        bn_grad_normalized = F.normalize(bn_grad.unsqueeze(0), dim=1)
                        ar_grad_normalized = F.normalize(ar_grad.unsqueeze(0), dim=1)

                        # Cosine similarity and raw conflict
                        cos_sim = F.cosine_similarity(bn_grad_normalized, ar_grad_normalized).item()
                        raw_conflict = 1 - cos_sim

                        # Compute Normalized Conflict Score (subtract noise floor)
                        ncs = conflict_monitor.compute_normalized_conflict(raw_conflict, layer)
                        layer_ncs[layer] = ncs

                        # Compute allocation signal (magnitude-weighted)
                        # For AR stage, AR is the "new" language, BN is the "existing" language
                        allocation_signal = conflict_monitor.compute_allocation_signal(
                            layer_idx=layer,
                            en_norm=bn_norm,  # BN acts as "existing" language in this stage
                            new_lang_norm=ar_norm,
                            normalized_conflict=ncs,
                            lambda_weight=lambda_weight
                        )
                        layer_allocation_signals[layer] = allocation_signal

                        # Update EMA trackers
                        conflict_monitor.update_ema_trackers(
                            layer_idx=layer,
                            raw_conflict=raw_conflict,
                            ncs=ncs,
                            en_norm=bn_norm,
                            new_lang_norm=ar_norm,
                            allocation_signal=allocation_signal
                        )

                        # Log diagnostic data
                        conflict_monitor.log_diagnostic(
                            step=step,
                            layer_idx=layer,
                            raw_conflict=raw_conflict,
                            ncs=ncs,
                            en_norm=bn_norm,
                            new_lang_norm=ar_norm,
                            allocation_signal=allocation_signal,
                            lambda_weight=lambda_weight
                        )

                        # Store comprehensive conflict metrics
                        conflict_history_ar[layer].append({
                            'step': step,
                            'cosine_similarity': cos_sim,
                            'conflict_score': raw_conflict,
                            'normalized_conflict_score': ncs,
                            'allocation_signal': allocation_signal,
                            'ar_grad_norm': ar_norm,
                            'bn_grad_norm': bn_norm,
                            'noise_floor': conflict_monitor.noise_floor.get(layer, 0.8),
                            'ema_ncs': conflict_monitor.ema_ncs[layer].value,
                            'ema_allocation': conflict_monitor.ema_allocation[layer].value,
                            'norm_ratio_ar_to_bn': ar_norm / (bn_norm + 1e-8),
                        })

                        layer_conflicts[layer] = raw_conflict

                # Find peak layer using SMOOTHED ALLOCATION SIGNALS (not raw conflict)
                if layer_allocation_signals:
                    # Use smoothed allocation signals for robust peak selection
                    smoothed_signals = conflict_monitor.get_smoothed_allocation_signals()

                    # For early steps when EMA isn't populated, use raw signals
                    if not smoothed_signals or all(v is None for v in smoothed_signals.values()):
                        smoothed_signals = layer_allocation_signals

                    peak_layer = conflict_monitor.select_peak_layer_robust(
                        allocation_signals=smoothed_signals,
                        min_threshold=0.01  # Lower threshold for early training
                    )

                    # Fallback to raw max if no valid peak
                    if peak_layer is None and layer_allocation_signals:
                        peak_layer = max(layer_allocation_signals, key=layer_allocation_signals.get)

                    if peak_layer is not None:
                        peak_allocation = smoothed_signals.get(peak_layer, layer_allocation_signals.get(peak_layer, 0))
                        max_ncs = max(layer_ncs.values()) if layer_ncs else 0

                        # Compute confidence as ratio of peak to second highest
                        sorted_signals = sorted(smoothed_signals.items(), key=lambda x: x[1] if x[1] else 0, reverse=True)
                        if len(sorted_signals) > 1 and sorted_signals[1][1]:
                            peak_confidence = sorted_signals[0][1] / (sorted_signals[1][1] + 1e-8)
                        else:
                            peak_confidence = 1.0

                        training_metrics_ar['peak_conflict_layers'].append(peak_layer)
                        training_metrics_ar['steps'].append(step)
                        training_metrics_ar['losses'].append(accumulated_loss * config.gradient_accumulation)

                        # Store additional metrics
                        if 'peak_confidence' not in training_metrics_ar:
                            training_metrics_ar['peak_confidence'] = []
                        training_metrics_ar['peak_confidence'].append(peak_confidence)

                        if 'peak_allocation_signal' not in training_metrics_ar:
                            training_metrics_ar['peak_allocation_signal'] = []
                        training_metrics_ar['peak_allocation_signal'].append(peak_allocation)

                        if 'max_ncs' not in training_metrics_ar:
                            training_metrics_ar['max_ncs'] = []
                        training_metrics_ar['max_ncs'].append(max_ncs)

                        accumulated_loss = 0

                        # Log with new metrics
                        progress_bar.set_postfix({
                            'loss': f"{training_metrics_ar['losses'][-1]:.4f}",
                            'peak_L': peak_layer,
                            'max_NCS': f"{max_ncs:.3f}",
                            'alloc': f"{peak_allocation:.2e}",
                            'conf': f"{peak_confidence:.2f}"
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
            if enable_checkpointing and step > 0 and step % config.checkpoint_interval == 0:
                checkpoint_manager.save_checkpoint(
                    stage='ar', step=step, model=model, optimizer=optimizer,
                    scheduler=scheduler, conflict_history=conflict_history_ar,
                    training_metrics=training_metrics_ar,
                    performance_tracking=performance_tracking
                )

        progress_bar.close()
        print("\n✓ Arabic training complete")

        # Report any skipped steps
        if training_metrics_ar['skipped_steps_oom']:
            print(f"⚠️  {len(training_metrics_ar['skipped_steps_oom'])} conflict logging steps skipped due to OOM")
            print(f"   Skipped at steps: {training_metrics_ar['skipped_steps_oom']}")

        # Evaluate after Arabic
        print("\n" + "="*80)
        print("EVALUATION AFTER ARABIC TRAINING")
        print("="*80)

        # Test set evaluations
        performance_tracking["after_ar"]["en_test"] = evaluate_on_test(
            model, en_test_loader, "English Test", device, tokenizer
        )
        performance_tracking["after_ar"]["bn_test"] = evaluate_on_test(
            model, bn_test_loader, "Bengali Test", device, tokenizer
        )
        performance_tracking["after_ar"]["ar_test"] = evaluate_on_test(
            model, ar_test_loader, "Arabic Test", device, tokenizer
        )
        # Training set evaluations (for true forgetting measurement)
        performance_tracking["after_ar"]["en_train"] = evaluate_on_test(
            model, en_train_eval_loader, "English Train", device, tokenizer
        )
        performance_tracking["after_ar"]["bn_train"] = evaluate_on_test(
            model, bn_train_eval_loader, "Bengali Train", device, tokenizer
        )
        performance_tracking["after_ar"]["ar_train"] = evaluate_on_test(
            model, ar_train_eval_loader, "Arabic Train", device, tokenizer
        )

        print(f"EN Test  - Loss: {performance_tracking['after_ar']['en_test']['loss']:.4f}, "
              f"Perplexity: {performance_tracking['after_ar']['en_test']['perplexity']:.2f}")
        print(f"EN Train - Loss: {performance_tracking['after_ar']['en_train']['loss']:.4f}, "
              f"Perplexity: {performance_tracking['after_ar']['en_train']['perplexity']:.2f}")
        print(f"BN Test  - Loss: {performance_tracking['after_ar']['bn_test']['loss']:.4f}, "
              f"Perplexity: {performance_tracking['after_ar']['bn_test']['perplexity']:.2f}")
        print(f"BN Train - Loss: {performance_tracking['after_ar']['bn_train']['loss']:.4f}, "
              f"Perplexity: {performance_tracking['after_ar']['bn_train']['perplexity']:.2f}")
        print(f"AR Test  - Loss: {performance_tracking['after_ar']['ar_test']['loss']:.4f}, "
              f"Perplexity: {performance_tracking['after_ar']['ar_test']['perplexity']:.2f}")
        print(f"AR Train - Loss: {performance_tracking['after_ar']['ar_train']['loss']:.4f}, "
              f"Perplexity: {performance_tracking['after_ar']['ar_train']['perplexity']:.2f}")

        # Save final checkpoint
        if enable_checkpointing:
            checkpoint_manager.save_checkpoint(
                stage='ar_complete', step=config.total_steps, model=model, optimizer=optimizer,
                scheduler=scheduler, conflict_history=conflict_history_ar,
                training_metrics=training_metrics_ar,
                performance_tracking=performance_tracking
            )
        else:
            print("⏭️  Skipping checkpoint save (checkpointing disabled)")

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

    # Data quality check
    bn_skipped = len(training_metrics_bn.get('skipped_steps_oom', []))
    bn_total_checkpoints = len(peak_layers_bn) + bn_skipped
    bn_data_quality = ((bn_total_checkpoints - bn_skipped) / bn_total_checkpoints * 100) if bn_total_checkpoints > 0 else 100

    print(f"\n📊 KEY FINDINGS (EN→BN):")
    print(f"  ├─ Data quality: {bn_data_quality:.0f}% ({bn_skipped} checkpoints skipped due to OOM)")
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

    # Data quality check
    ar_skipped = len(training_metrics_ar.get('skipped_steps_oom', []))
    ar_total_checkpoints = len(peak_layers_ar) + ar_skipped
    ar_data_quality = ((ar_total_checkpoints - ar_skipped) / ar_total_checkpoints * 100) if ar_total_checkpoints > 0 else 100

    print(f"\n📊 KEY FINDINGS (BN→AR):")
    print(f"  ├─ Data quality: {ar_data_quality:.0f}% ({ar_skipped} checkpoints skipped due to OOM)")
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
    # CATASTROPHIC FORGETTING ANALYSIS (Using TRAINING sets for TRUE forgetting)
    # ========================================================================
    print("\n" + "="*80)
    print("🔬 CATASTROPHIC FORGETTING ANALYSIS")
    print("="*80)
    print("Note: TRUE forgetting = degradation on TRAINING data (not just test data)")

    # Check for training set data (preferred) or fall back to test data
    has_en_train = 'after_en' in performance_tracking and 'en_train' in performance_tracking.get('after_en', {})
    has_bn_train = 'after_bn' in performance_tracking and 'bn_train' in performance_tracking.get('after_bn', {})

    train_degradations = []
    test_degradations = []

    # English forgetting analysis
    if has_en_train and 'after_bn' in performance_tracking and 'en_train' in performance_tracking['after_bn']:
        en_train_initial = performance_tracking['after_en']['en_train']['perplexity']
        en_train_after_bn = performance_tracking['after_bn']['en_train']['perplexity']
        en_train_bn_deg = ((en_train_after_bn - en_train_initial) / en_train_initial) * 100

        en_test_initial = performance_tracking['after_en']['en_test']['perplexity']
        en_test_after_bn = performance_tracking['after_bn']['en_test']['perplexity']
        en_test_bn_deg = ((en_test_after_bn - en_test_initial) / en_test_initial) * 100

        print(f"\n📉 ENGLISH FORGETTING (EN→BN):")
        print(f"  Train: {en_train_initial:.2f} → {en_train_after_bn:.2f} ({en_train_bn_deg:+.1f}%) [TRUE FORGETTING]")
        print(f"  Test:  {en_test_initial:.2f} → {en_test_after_bn:.2f} ({en_test_bn_deg:+.1f}%) [generalization]")

        if 'after_ar' in performance_tracking and 'en_train' in performance_tracking['after_ar']:
            en_train_final = performance_tracking['after_ar']['en_train']['perplexity']
            en_train_total_deg = ((en_train_final - en_train_initial) / en_train_initial) * 100
            en_test_final = performance_tracking['after_ar']['en_test']['perplexity']
            en_test_total_deg = ((en_test_final - en_test_initial) / en_test_initial) * 100

            print(f"  After AR Train: {en_train_final:.2f} ({en_train_total_deg:+.1f}% total)")
            print(f"  After AR Test:  {en_test_final:.2f} ({en_test_total_deg:+.1f}% total)")
            train_degradations.append(en_train_total_deg)
            test_degradations.append(en_test_total_deg)

    # Bengali forgetting analysis
    if has_bn_train and 'after_ar' in performance_tracking and 'bn_train' in performance_tracking['after_ar']:
        bn_train_initial = performance_tracking['after_bn']['bn_train']['perplexity']
        bn_train_final = performance_tracking['after_ar']['bn_train']['perplexity']
        bn_train_deg = ((bn_train_final - bn_train_initial) / bn_train_initial) * 100

        bn_test_initial = performance_tracking['after_bn']['bn_test']['perplexity']
        bn_test_final = performance_tracking['after_ar']['bn_test']['perplexity']
        bn_test_deg = ((bn_test_final - bn_test_initial) / bn_test_initial) * 100

        print(f"\n📉 BENGALI FORGETTING (BN→AR):")
        print(f"  Train: {bn_train_initial:.2f} → {bn_train_final:.2f} ({bn_train_deg:+.1f}%) [TRUE FORGETTING]")
        print(f"  Test:  {bn_test_initial:.2f} → {bn_test_final:.2f} ({bn_test_deg:+.1f}%) [generalization]")
        train_degradations.append(bn_train_deg)
        test_degradations.append(bn_test_deg)

    # Summary
    if train_degradations:
        avg_train_deg = sum(train_degradations) / len(train_degradations)
        avg_test_deg = sum(test_degradations) / len(test_degradations) if test_degradations else 0

        print(f"\n🎯 BACKWARD TRANSFER (BWT) SUMMARY:")
        print(f"  Average TRAIN forgetting: {avg_train_deg:.1f}% (TRUE catastrophic forgetting)")
        print(f"  Average TEST degradation: {avg_test_deg:.1f}% (generalization loss)")

        if avg_train_deg > 10:
            print(f"\n✅ CATASTROPHIC FORGETTING CONFIRMED (>10% train degradation)")
            print(f"   → Model is overwriting previously learned knowledge")
            print(f"   → This validates the need for dynamic expert allocation")
        elif avg_train_deg > 5:
            print(f"\n⚠️  MODERATE FORGETTING (5-10% train degradation)")
            print(f"   → Some knowledge overwriting occurring")
        else:
            print(f"\n✅ MINIMAL FORGETTING (<5% train degradation)")
            print(f"   → Model retains previous knowledge well")
    else:
        print(f"\n⚠️  Insufficient data for forgetting analysis (stages may have been skipped)")

    # ========================================================================
    # EXPORT RESULTS
    # ========================================================================
    print("\n" + "="*80)
    print("EXPORTING RESULTS")
    print("="*80)

    # Helper to safely convert values for JSON
    def safe_float(v):
        if v is None:
            return None
        if isinstance(v, (int, float, np.number)):
            return float(v)
        return v

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

        # Noise floor calibration data (from EN-EN sanity check)
        'noise_floor_calibration': {
            'calibrated': conflict_monitor.noise_calibrated,
            'noise_floor_per_layer': {str(k): float(v) for k, v in conflict_monitor.noise_floor.items()},
            'average_noise_floor': float(np.mean(list(conflict_monitor.noise_floor.values()))),
            'en_en_conflicts': {str(k): float(v) for k, v in en_en_conflicts.items()} if en_en_conflicts else {},
            'validation': conflict_monitor.get_validation_checks()
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
                    {k: safe_float(v) if not isinstance(v, int) else int(v)
                     for k, v in h.items()}
                    for h in history
                ]
                for layer, history in conflict_history_bn.items()
            },
            'training_metrics': {
                'steps': [int(s) for s in training_metrics_bn['steps']],
                'losses': [float(l) for l in training_metrics_bn['losses']],
                'peak_conflict_layers': [int(l) for l in training_metrics_bn['peak_conflict_layers']],
                'skipped_steps_oom': [int(s) for s in training_metrics_bn.get('skipped_steps_oom', [])],
                'data_quality_pct': float(bn_data_quality),
                # New NCS and allocation signal metrics
                'peak_allocation_signal': [float(s) for s in training_metrics_bn.get('peak_allocation_signal', [])],
                'max_ncs': [float(n) for n in training_metrics_bn.get('max_ncs', [])],
                'peak_confidence': [float(c) for c in training_metrics_bn.get('peak_confidence', [])]
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
                    {k: safe_float(v) if not isinstance(v, int) else int(v)
                     for k, v in h.items()}
                    for h in history
                ]
                for layer, history in conflict_history_ar.items()
            },
            'training_metrics': {
                'steps': [int(s) for s in training_metrics_ar['steps']],
                'losses': [float(l) for l in training_metrics_ar['losses']],
                'peak_conflict_layers': [int(l) for l in training_metrics_ar['peak_conflict_layers']],
                'skipped_steps_oom': [int(s) for s in training_metrics_ar.get('skipped_steps_oom', [])],
                'data_quality_pct': float(ar_data_quality),
                # New NCS and allocation signal metrics
                'peak_allocation_signal': [float(s) for s in training_metrics_ar.get('peak_allocation_signal', [])],
                'max_ncs': [float(n) for n in training_metrics_ar.get('max_ncs', [])],
                'peak_confidence': [float(c) for c in training_metrics_ar.get('peak_confidence', [])]
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
        },

        # Full diagnostic history from ConflictMonitor
        'diagnostic_history': conflict_monitor.diagnostic_history
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

    # Export detailed diagnostics for post-hoc analysis
    diagnostics_path = results_dir / 'conflict_diagnostics.json'
    diagnostics_data = {
        'noise_floor': {str(k): float(v) for k, v in conflict_monitor.noise_floor.items()},
        'calibrated': conflict_monitor.noise_calibrated,
        'diagnostic_history': conflict_monitor.diagnostic_history,
        'summary': {
            'total_diagnostic_entries': len(conflict_monitor.diagnostic_history),
            'average_noise_floor': float(np.mean(list(conflict_monitor.noise_floor.values()))),
        }
    }
    with open(diagnostics_path, 'w') as f:
        json.dump(diagnostics_data, f, indent=2)
    print(f"✓ Conflict diagnostics saved to: {diagnostics_path}")

    vram_monitor.summary()
    print(f"\n🎉 Training complete! Checkpoints in: {config.checkpoint_dir}")
    print(f"   Results in: {config.results_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CADEA Local Training")
    parser.add_argument("--test", action="store_true", help="Run quick test (5 min)")
    parser.add_argument("--stage", choices=['en', 'bn', 'ar', 'full'], default='full',
                       help="Training stage to run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--checkpointing", choices=['yes', 'no'], default='yes',
                       help="Enable/disable checkpointing (use 'no' for Kaggle/Colab)")
    parser.add_argument("--en-samples", type=int, default=None,
                       help="Number of English samples (overrides config)")
    parser.add_argument("--bn-samples", type=int, default=None,
                       help="Number of Bengali samples (overrides config)")
    parser.add_argument("--ar-samples", type=int, default=None,
                       help="Number of Arabic samples (overrides config)")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Batch size for training (overrides config)")
    args = parser.parse_args()

    main(args)
