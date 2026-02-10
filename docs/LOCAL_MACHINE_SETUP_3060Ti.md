# Running CADEA on Local Machine (3060 Ti - 8GB VRAM)
# ======================================================

## ✅ YES, YOUR 3060 Ti CAN HANDLE THIS!

**Your specs:**
- GPU: RTX 3060 Ti (8GB VRAM)
- Model: Llama 3.2-1B (1.2B parameters)
- Required VRAM: ~6-7GB with optimizations

**Verdict: ✅ FEASIBLE with proper settings**

---

## 🎯 OPTIMAL CONFIGURATION FOR 3060 Ti

### Memory Budget Breakdown
```
Model weights (bfloat16):     ~2.4 GB
Optimizer states (AdamW):     ~4.8 GB
Gradients:                    ~2.4 GB
Activations (batch_size=1):   ~0.5 GB
-------------------------------------------
Total WITHOUT tricks:         ~10.1 GB  ❌ Too much!

WITH optimizations:
- Gradient checkpointing:     -1.5 GB
- Batch size = 1:             -0.3 GB
- Gradient accumulation:      Free!
-------------------------------------------
Total WITH optimizations:     ~6.5 GB  ✅ Fits!
```

---

## 📋 STEP-BY-STEP SETUP

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv cadea_env
source cadea_env/bin/activate  # Linux/Mac
# OR
cadea_env\Scripts\activate  # Windows

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install transformers==4.46.3 datasets accelerate sentencepiece protobuf
pip install matplotlib tqdm
```

### 2. Create Config for 3060 Ti

Create `config_local.py`:

```python
class Config:
    # Model
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    
    # Datasets (START SMALL TO TEST!)
    en_dataset = "tatsu-lab/alpaca"
    en_samples = 100
    bn_dataset = "md-nishat-008/Bangla-Instruct"
    bn_samples = 2000  # Reduced from 5000
    ar_dataset = "arbml/CIDAR"
    ar_samples = 2000  # Reduced from 5000
    
    # ⚠️ CRITICAL: Memory-optimized settings for 3060 Ti
    batch_size = 1  # Reduced from 2
    gradient_accumulation = 4  # Increased from 2 (same effective batch size)
    
    # Training
    learning_rate = 5e-6
    layers_to_track = [0, 3, 6, 9, 12, 15]
    log_interval = 50
    checkpoint_interval = 250  # More frequent saves
    total_steps = 1500  # Slightly reduced for faster iteration
    
    max_length = 512
    use_bf16 = True  # Your 3060 Ti supports bfloat16!
    gradient_checkpointing = True  # ESSENTIAL for 8GB VRAM
    
    # Paths
    checkpoint_dir = "./checkpoints"  # Local directory
    hf_token = "your_token_here"  # Or set via env variable
```

### 3. Verify GPU Setup

Create `test_gpu.py`:

```python
import torch
import transformers

print("="*60)
print("GPU VERIFICATION")
print("="*60)

# Check CUDA
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Test bfloat16 support
    try:
        x = torch.randn(100, 100, dtype=torch.bfloat16, device='cuda')
        print(f"✅ bfloat16 supported")
    except:
        print(f"❌ bfloat16 NOT supported (will use float16)")
else:
    print("❌ CUDA not available! Check your PyTorch installation.")

print("="*60)
```

Run it:
```bash
python test_gpu.py
```

Expected output:
```
============================================================
GPU VERIFICATION
============================================================
PyTorch version: 2.x.x+cu118
CUDA available: True
CUDA version: 11.8
GPU: NVIDIA GeForce RTX 3060 Ti
Total VRAM: 8.00 GB
✅ bfloat16 supported
============================================================
```

---

## 🚀 RUNNING THE EXPERIMENT

### Option 1: Quick Test (5 minutes)

Test everything works before full run:

```python
# test_config.py
class Config:
    # Use TINY datasets for testing
    en_samples = 10
    bn_samples = 50
    ar_samples = 50
    
    batch_size = 1
    gradient_accumulation = 2
    total_steps = 100  # Just 100 steps to test
    checkpoint_interval = 50
    
    # Everything else same as full config
```

```bash
python cadea_training_local.py --config test_config
```

This should complete in ~5 minutes and verify:
- ✅ Model loads successfully
- ✅ Training doesn't OOM
- ✅ Checkpoints save correctly
- ✅ Evaluation works

### Option 2: Full Experiment (8-12 hours)

```bash
# Run with monitoring
python cadea_training_local.py --config config_local

# OR with nohup (runs in background)
nohup python cadea_training_local.py --config config_local > training.log 2>&1 &

# Monitor progress
tail -f training.log
```

**Estimated time on 3060 Ti:**
- English training: ~30 minutes
- Bengali training: ~3-4 hours
- Arabic training: ~3-4 hours
- **Total: ~8-10 hours**

---

## 💾 MEMORY MONITORING

### Watch VRAM Usage (Linux)

```bash
# Terminal 1: Run training
python cadea_training_local.py

# Terminal 2: Monitor VRAM
watch -n 1 nvidia-smi
```

### Watch VRAM Usage (Windows)

Open Task Manager → Performance → GPU 0 → Dedicated GPU Memory

### In Python Script

Add this to your training loop:

```python
if step % 100 == 0:
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"VRAM: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
```

**Expected VRAM usage:**
- Model loading: ~2.5 GB
- Training: ~6.5-7.0 GB
- If you see >7.5 GB: ⚠️ Reduce batch_size or max_length

---

## 🔧 TROUBLESHOOTING

### "CUDA out of memory"

**Solution 1: Reduce batch size**
```python
config.batch_size = 1  # Already at minimum
config.gradient_accumulation = 8  # Increase this instead
```

**Solution 2: Reduce sequence length**
```python
config.max_length = 256  # Down from 512
```

**Solution 3: Use smaller model**
```python
config.model_name = "meta-llama/Llama-3.2-1B"  # Base model (no instruct)
# OR
config.model_name = "microsoft/phi-2"  # 2.7B but more efficient
```

**Solution 4: Reduce layers tracked**
```python
config.layers_to_track = [0, 6, 12]  # Track fewer layers
```

### "Training is very slow"

This is normal! With batch_size=1, training is inherently slower than Kaggle's larger GPUs.

**Speed comparisons:**
- Kaggle P100 (batch_size=2): ~2 steps/sec
- Your 3060 Ti (batch_size=1): ~0.5-1 steps/sec
- **Tradeoff: You control it, no timeouts!**

**Speedup tricks:**
```python
# 1. Reduce sequence length
config.max_length = 256  # 2x faster

# 2. Reduce dataset size
config.bn_samples = 1000  # Still scientifically valid
config.ar_samples = 1000

# 3. Reduce total steps
config.total_steps = 1000  # Still shows conflict migration
```

### "bfloat16 not supported"

If your 3060 Ti doesn't support bfloat16 (rare), use float16:

```python
config.use_bf16 = False
# Model will auto-fall back to float16
```

---

## 📊 ADVANTAGES OF LOCAL MACHINE

### ✅ Pros
1. **No timeouts** - Run for days if needed
2. **Full control** - Pause/resume anytime
3. **Faster iteration** - No kernel restart delays
4. **Keep checkpoints** - Unlimited storage
5. **Debug easily** - Use your IDE, debugger, etc.

### ❌ Cons
1. **Slower per-step** - Smaller batch size
2. **Ties up GPU** - Can't game while training
3. **Electricity cost** - ~8-10 hours at 200W
4. **Need supervision** - Check for crashes

---

## 🎯 RECOMMENDED WORKFLOW

### Day 1: Quick Test
```bash
# 5-minute test run
python cadea_training_local.py --test
```
**Goal:** Verify everything works, no OOM errors

### Day 2: English Stage
```bash
# Run just English training (~30 min)
python cadea_training_local.py --stage en
```
**Goal:** Get English baseline, verify metrics look good

### Day 3: Full Run
```bash
# Run complete experiment (leave overnight)
nohup python cadea_training_local.py --full > training.log 2>&1 &
```
**Goal:** Complete EN→BN→AR sequence

### Day 4: Analysis
```bash
# Generate plots and results
python analyze_results.py
```
**Goal:** Create thesis figures

---

## 💡 PRO TIPS FOR 3060 Ti

1. **Close all other programs** - Give your GPU full focus
2. **Monitor temperature** - 3060 Ti should stay <80°C
3. **Use `watch nvidia-smi`** - Catch OOM issues early
4. **Start with test config** - Don't waste hours on full run if setup is wrong
5. **Save often** - `checkpoint_interval = 250` is good for 3060 Ti
6. **Run overnight** - 8-10 hours is perfect for sleep time!

---

## 📈 EXPECTED PERFORMANCE

**3060 Ti vs Kaggle P100:**

| Metric | 3060 Ti (8GB) | Kaggle P100 (16GB) |
|--------|---------------|-------------------|
| Batch size | 1 | 2 |
| Steps/sec | ~0.5-1 | ~2 |
| Total time | ~10 hours | ~4 hours |
| Cost | $0 (your GPU) | $0 (free tier) |
| Control | ✅ Full | ❌ Timeouts |
| Reliability | ✅ High | ⚠️ Session limits |

**Verdict:** 3060 Ti is 2-3x slower but **more reliable** for thesis work!

---

## 🎓 FOR YOUR THESIS TIMELINE

**2.5 months = 10 weeks**

Suggested schedule:
- **Week 1:** Setup + test runs (this guide)
- **Week 2-3:** Full experiment + variations
- **Week 4-5:** Analysis + visualizations
- **Week 6-8:** Writing
- **Week 9-10:** Revisions + buffer

**Your 3060 Ti is perfect for this!** You can run 2-3 full experiments in Week 2-3.

---

## 🆘 NEED HELP?

If you hit issues:

1. **Check VRAM:** `nvidia-smi` should show <7.5 GB used
2. **Check logs:** Look for "CUDA out of memory" or NaN losses
3. **Test minimal config:** Run with en_samples=10 first
4. **Share error:** Post the full traceback

**Common issues (95% of problems):**
- Forgot `gradient_checkpointing = True`
- Batch size too large
- Multiple Python processes using GPU
- CUDA driver outdated

---

## ✅ FINAL CHECKLIST

Before starting full experiment:

- [ ] GPU verified with `test_gpu.py`
- [ ] Config set to `batch_size=1`, `gradient_checkpointing=True`
- [ ] Test run completed successfully (10 samples)
- [ ] Checkpoint directory has write permissions
- [ ] Hugging Face token set (if needed)
- [ ] Enough disk space (~20 GB for checkpoints)
- [ ] Other programs closed (give GPU full power)

**You're ready! Your 3060 Ti can absolutely handle this. Go validate that thesis! 🚀**
