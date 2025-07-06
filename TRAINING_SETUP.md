# REBEL Model Training Setup Guide

This guide provides step-by-step instructions for setting up and training the REBEL model for MaintIE relation extraction, based on successfully training the model from scratch.

## 🎯 Overview

This guide covers the **practical setup process** for training REBEL on the MaintIE dataset. It includes:

- ✅ **Exact dependency versions** that work
- ✅ **Environment setup** with conda
- ✅ **Training configuration** with working parameters
- ✅ **Troubleshooting** for common issues
- ✅ **Monitoring and evaluation** tools

For research experiment details, see [MODELS.md](./MODELS.md).

## 📋 Prerequisites

- **Python 3.11** (tested with 3.11.11)
- **Linux environment** (tested on Ubuntu/Linux 6.8.0)
- **CPU training** (GPU optional but faster)
- **~10GB free disk space** for model checkpoints

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda create -n mining_analytics python=3.11
conda activate mining_analytics

# Install exact working dependencies
pip install -r requirements.txt
```

### 2. Run Training

```bash
# Activate environment (critical!)
conda activate mining_analytics

# Run full training
python run_full_training_final.py 2>&1 | tee training_log.log
```

### 3. Monitor Progress

```bash
# In another terminal
python monitor_training.py
```

## 🔧 Detailed Setup

### Environment Dependencies

The `requirements.txt` file includes **exact versions** that work together:

**Core Training Dependencies:**

```
torch==2.6.0+cu124
transformers==4.35.0
datasets==2.14.0
pytorch-lightning==2.1.0
```

**Configuration & Logging:**

```
hydra-core==1.3.2
omegaconf==2.3.0
wandb==0.15.12
```

**Critical Version Notes:**

- `fsspec==2023.4.0` (must be this exact version for datasets compatibility)
- `transformers==4.35.0` (4.34.0 causes issues with newer PyTorch)
- `datasets==2.14.0` (2.14.5 has breaking changes)

### Training Configuration

The working configuration (`run_full_training_final.py`) includes:

```python
# Training parameters
max_epochs = 3
max_steps = 500
batch_size = 2
learning_rate = 3e-5
lr_scheduler = "linear"

# Data parameters
max_train_samples = None  # Use all 860 samples
max_val_samples = None   # Use all 108 samples
max_source_length = 256  # 8x longer than minimal setup
max_target_length = 256

# Hardware settings
dataloader_num_workers = 1
dataloader_pin_memory = True
dataloader_drop_last = False
```

## 📊 Training Process

### Expected Performance

**Training Scale:**

- **Training samples**: 860 (full MAINTIE dataset)
- **Validation samples**: 108
- **Training steps**: 430 steps/epoch × 3 epochs = 1,290 total steps
- **Model size**: 406M parameters (REBEL-large)

**Performance Targets:**

- **Current minimal baseline**: F1 ≈ 15-25% (2 steps training)
- **Expected with full training**: F1 ≈ 45-60%
- **Paper benchmark**: F1 = 67-71% (MAINTIE Fine-Grained)

### Training Timeline

**CPU Training:**

- **Duration**: ~4-6 hours
- **Speed**: ~0.75 iterations/second
- **Checkpoint saving**: Every epoch + final model
- **Total checkpoints**: ~5-10GB

**GPU Training (if available):**

- **Duration**: ~1-2 hours
- **Speed**: ~3-5 iterations/second

### Monitoring Training

Use the monitoring script to track progress:

```bash
# Monitor training metrics
python monitor_training.py

# View training log
tail -f training_log.log

# Check checkpoint sizes
ls -lh experiments/rebel_final_corrected/
```

## 🔍 Troubleshooting

### Common Issues & Solutions

#### 1. ImportError: No module named 'hydra'

```bash
pip install hydra-core==1.3.2 omegaconf==2.3.0
```

#### 2. ImportError: No module named 'pytorch_lightning'

```bash
pip install pytorch-lightning==2.1.0
```

#### 3. IndexError: Index X out of range for dataset of size Y

**Cause**: Requesting more samples than available in dataset
**Solution**: Set `max_train_samples = None` and `max_val_samples = None`

#### 4. Key 'config' is not in struct

**Cause**: Hydra/PyTorch Lightning version conflict
**Solution**: Use exact versions from requirements.txt

#### 5. "conda activate mining_analytics" not working

**Cause**: Training script not using correct environment
**Solution**: Always run with explicit conda activation:

```bash
conda activate mining_analytics && python run_full_training_final.py
```

### Dependency Resolution

If you encounter version conflicts:

1. **Start fresh:**

```bash
conda deactivate
conda remove -n mining_analytics --all
conda create -n mining_analytics python=3.11
conda activate mining_analytics
```

2. **Install in order:**

```bash
pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.35.0
pip install datasets==2.14.0
pip install pytorch-lightning==2.1.0
pip install hydra-core==1.3.2
pip install omegaconf==2.3.0
pip install wandb==0.15.12
pip install fsspec==2023.4.0
# ... other dependencies
```

## 📈 Evaluation & Testing

### Test Trained Model

```bash
# Test model on sample mining texts
python test_simple_model.py

# Expected output:
# Input: "The crusher processes ore from the open pit mine"
# Output: Relations detected with confidence scores
```

### Evaluation Metrics

The training automatically tracks:

- **Precision, Recall, F1** (micro and macro)
- **Per-relation performance** (is a, contains, has part, etc.)
- **Loss curves** (training and validation)

### Model Checkpoints

Training saves multiple checkpoints:

- `last.ckpt` - Final model state
- `last-v1.ckpt` - Previous version
- `last-v2.ckpt` - Earlier version

## 🎯 Next Steps

After successful training:

1. **Evaluate on test set**: Use trained model on `maintie_test.json`
2. **Deploy to Azure ML**: Use Azure ML SDK v2 integration
3. **Production inference**: Extract relations from real mining documents

## 📚 Additional Resources

- **Paper**: [MaintIE: A Fine-Grained Annotation Schema and Benchmark](https://aclanthology.org/2024.lrec-main.954.pdf)
- **Original REBEL**: [Babelscape/rebel](https://github.com/Babelscape/rebel)
- **Research experiments**: [MODELS.md](./MODELS.md)

## 🐛 Getting Help

If you encounter issues:

1. **Check versions**: Ensure exact dependency versions from requirements.txt
2. **Verify environment**: `conda activate mining_analytics` before training
3. **Review logs**: Check training logs for specific error messages
4. **Clean setup**: Remove conda environment and start fresh if needed

## 🏆 Success Indicators

You'll know the setup is working when:

- ✅ Training starts without import errors
- ✅ Model loads successfully (406M parameters)
- ✅ Training progresses (0.7+ it/s)
- ✅ Validation runs without errors
- ✅ Checkpoints are saved to experiments/

---

_Last updated: Based on successful training run with 860 samples, 500 steps, achieving stable training at ~0.75 it/s_
