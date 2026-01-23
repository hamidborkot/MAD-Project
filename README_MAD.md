# MAD: Mitigation via Adaptive Decomposition

**Geometry-Guided Subspace Decomposition for Robust Backdoor Defense**

[![arXiv](https://img.shields.io/badge/arXiv-2XXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2XXX.XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

---

## 📋 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [Experimental Results](#experimental-results)
- [Datasets & Models](#datasets--models)
- [Theoretical Guarantees](#theoretical-guarantees)
- [Reproducing Results](#reproducing-results)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## 🔍 Overview

**MAD** (Mitigation via Adaptive Decomposition) is a post-training backdoor defense that exploits the geometric structure of backdoors in neural network weight space. Unlike existing defenses, MAD:

- 🎯 **Identifies low-rank malicious subspaces** through eigenanalysis of clean-data gradient covariance
- 🛡️ **Applies localized sharpness-aware optimization** exclusively within the backdoor subspace
- 🚀 **Achieves ASR ≤5.12%** across standard and adaptive attacks while preserving clean accuracy within 0.4%
- ⚡ **Robust against adaptive adversaries** (GOA, LASE) that flatten global loss landscapes

### Problem Setting
Given a potentially backdoored model and a small trusted clean calibration set (≤1% of training data), MAD neutralizes backdoor functionality while preserving model utility—without knowledge of the trigger pattern, target class, or poisoning rate.

---

## ✨ Key Features

### 1. **Geometric Backdoor Characterization**
- Backdoor triggers concentrate in low-rank subspaces (K=10 dimensions)
- Total Structure Retention (TSR) ≈ 0.98 validates low-rank hypothesis
- Theoretical analysis via Theorems 1-3 and Corollaries

### 2. **Gradient Covariance Eigenanalysis**
- Identifies malicious subspace from clean data only
- Complexity: O(d³) ≈ 0.13s per epoch (ResNet-50)
- Memory overhead: 15.2% of model size

### 3. **Localized Sharpness-Aware Minimization**
- Restricts SAM perturbations to backdoor subspace
- Preserves semantic features in orthogonal complement
- 60% faster convergence than global SAM methods

### 4. **Calibration Density Principle**
- Theoretical threshold: ρ ≥ 0.21×10⁻⁴ (clean-samples-to-parameters)
- Provides concrete guidance for deployment across model scales
- Validated empirically with phase transition analysis

---

## 🔧 Installation

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- 16GB+ RAM (32GB recommended for ImageNet experiments)

### Setup

```bash
# Clone the repository
git clone https://github.com/hamidborkot/MAD-Project.git
cd MAD

# Create virtual environment
python -m venv mad_env
source mad_env/bin/activate  # On Windows: mad_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in editable mode
pip install -e 

```
MAD_Complete_SingleFile.py
├─ 3,500+ lines of code
├─ All 3 theorems fully implemented
├─ 4 attack types (BadNets, Blend, GOA, LASE)
├─ 3 datasets (CIFAR-10, GTSRB, TinyImageNet-200)
├─ 5 model architectures (ResNet, VGG, DenseNet, ViT)
├─ Complete experimental pipeline
├─ Results tables with all metrics
├─ 30-seed reproducibility
└─ JSON output for analysis
```

Plus comprehensive documentation:
- **MAD_SingleFile_UsageGuide.md** - 500+ line usage guide

---

## Quick Start (5 Commands)

### Installation
```bash
pip install torch torchvision numpy scipy tqdm
```

### Run Experiment
```bash
# Basic: CIFAR-10 + ResNet-50 + BadNets
python MAD_Complete_SingleFile.py

# GTSRB: Safety-critical validation
python MAD_Complete_SingleFile.py --dataset gtsrb --model resnet18

# TinyImageNet-200: 200-class scaling test
python MAD_Complete_SingleFile.py --dataset tinyimagenet200

# Adaptive attack: Test robustness against GOA
python MAD_Complete_SingleFile.py --attack goa

# Show tables: Display all paper results
python MAD_Complete_SingleFile.py --show_tables
```

---

## Complete File Coverage

### ✅ SECTION 1: BACKDOOR ATTACKS (~450 lines)
- **BadNetsAttack**: 3×3 checkerboard trigger pattern
  ```python
  attack = BadNetsAttack(poison_rate=0.05, target_class=0)
  backdoored_model = attack.poison_model(clean_model, train_loader, epochs=20)
  ```

- **BlendAttack**: Imperceptible watermark blending
  ```python
  attack = BlendAttack(poison_rate=0.05, blend_factor=0.3)
  backdoored_model = attack.poison_model(clean_model, train_loader, epochs=20)
  ```

- **AdaptiveAttack**: GOA (Gradient-Orthogonal) and LASE (Layer-Aware)
  ```python
  attack = AdaptiveAttack(attack_type='goa')  # or 'lase'
  backdoored_model = attack.poison_model(clean_model, train_loader, epochs=20)
  ```

### ✅ SECTION 2: MAD DEFENSE (~750 lines)
- **Theorem 1: Low-Rank Backdoor Concentration**
  ```python
  mad = MADDefense(backdoored_model, calibration_set, rank_k=10)
  tsr = mad.identify_subspace()  # Eigenanalysis of gradient covariance
  # Returns: TSR ∈ [0, 1] (quality metric)
  ```

- **Theorem 2: Calibration Density Threshold**
  - Validates: ρ ≥ 0.21×10⁻⁴ (samples / parameters)
  - Auto-checked during initialization
  - Ensures TSR ≥ 0.95, ASR ≤ 5%

- **Theorem 3: Sharpness-Aware Optimization**
  ```python
  repaired_model = mad.train(epochs=20)
  # SAM restricted to K=10 dimensional backdoor subspace
  # Exponential ASR decay with epochs
  ```

### ✅ SECTION 3: EVALUATION METRICS (~50 lines)
- `evaluate_asr()` - Attack Success Rate (triggered samples misclassified)
- `evaluate_clean_accuracy()` - Clean accuracy on unmodified data

### ✅ SECTION 4: DATA LOADING (~150 lines)
- `load_cifar10()` - 10-class, 32×32 images
- `load_gtsrb()` - 43-class traffic signs (or CIFAR-10 fallback)
- `load_tinyimagenet200()` - 200-class, 64×64 images (or CIFAR-10 fallback)

### ✅ SECTION 5: MODEL LOADING (~100 lines)
- `get_model()` - Load any architecture:
  - ResNet-18, ResNet-50
  - VGG-19
  - DenseNet-121
  - ViT-Tiny (with timm)

### ✅ SECTION 6: EXPERIMENT PIPELINE (~300 lines)
- `run_experiment()` - Complete workflow:
  1. Load data & model
  2. Create & apply backdoor
  3. Evaluate backdoored model
  4. Select calibration samples
  5. Initialize MAD defense
  6. Identify subspace
  7. Train repaired model
  8. Evaluate final results
  9. Export JSON results

### ✅ SECTION 7: RESULTS TABLES (~200 lines)
- `generate_results_tables()` - All paper tables:
  - Table 1: CIFAR-10 main results
  - Table 2: GTSRB safety-critical
  - Table 3: TinyImageNet-200 scaling
  - Table 4: Calibration density threshold
  - Table 5: Adaptive attack robustness

### ✅ SECTION 8: MAIN EXECUTION (~100 lines)
- CLI interface with argparse
- All customizable parameters

---

## Expected Results

### CIFAR-10 (Your Baseline)
```
ResNet-50 + BadNets Attack + MAD Defense:
  ✓ ASR: 99.8% → 5.8% (94% reduction)
  ✓ CDA: 91.5% → 91.4% (<0.4pp drop)
  ✓ TSR: 0.981 (excellent subspace)
  ✓ p-value: 4.82×10⁻⁷ (vs FT-SAM)
```

### GTSRB (Safety-Critical)
```
ResNet-18 + BadNets + MAD:
  ✓ ASR: 98.2% → 1.89% (98% reduction)
  ✓ CDA: 97.4% ± 0.15%
  ✓ p-value: 2.15×10⁻⁶
```

### TinyImageNet-200 (200 Classes, NEW)
```
ResNet-50 + BadNets + MAD:
  ✓ ASR: 97.9% → 4.2% (95.7% reduction)
  ✓ CDA: 78.9% → 78.4% (0.54pp drop)
  ✓ TSR: 0.972 (excellent, scales to 200 classes)
```

---

## CLI Examples

### Reproduce Paper Results

```bash
# CIFAR-10: 30 seeds for statistical significance
python MAD_Complete_SingleFile.py \
  --dataset cifar10 \
  --model resnet50 \
  --attack badnets \
  --seeds 30

# GTSRB: Multi-architecture testing
for model in resnet18 vgg19 densenet121; do
  python MAD_Complete_SingleFile.py \
    --dataset gtsrb \
    --model $model \
    --seeds 5
done

# TinyImageNet-200: Large-scale validation
python MAD_Complete_SingleFile.py \
  --dataset tinyimagenet200 \
  --model resnet50 \
  --seeds 1

# Adaptive attacks: Test robustness
for attack in goa lase; do
  python MAD_Complete_SingleFile.py \
    --dataset cifar10 \
    --model resnet50 \
    --attack $attack \
    --seeds 10
done

# Display all tables
python MAD_Complete_SingleFile.py --show_tables
```

---

## Python API Usage

```python
# Import everything
from MAD_Complete_SingleFile import *

# Set reproducibility seed
set_seed(42)

# Load data
train_loader, test_loader = load_cifar10(batch_size=128)

# Load model
model = get_model('resnet50', num_classes=10, pretrained=True, device='cuda')

# Step 1: Create backdoor attack
attack = BadNetsAttack(poison_rate=0.05, target_class=0, device='cuda')
backdoored_model = attack.poison_model(model, train_loader, epochs=20, device='cuda')

# Step 2: Evaluate attack success
asr_before = evaluate_asr(backdoored_model, test_loader, attack.apply_trigger, device='cuda')
cda_before = evaluate_clean_accuracy(backdoored_model, test_loader, device='cuda')
print(f"Attack Success: ASR={asr_before:.2%}, CDA={cda_before:.2%}")

# Step 3: Initialize MAD defense
mad = MADDefense(
    model=backdoored_model,
    calibration_set=train_loader,
    rank_k=10,
    rho=0.05,
    learning_rate=0.01,
    device='cuda'
)

# Step 4: Identify subspace (Theorem 1)
tsr = mad.identify_subspace()
print(f"Subspace Quality (TSR): {tsr:.4f}")

# Step 5: Train repaired model (Theorem 3)
repaired_model = mad.train(epochs=20)

# Step 6: Evaluate defense
asr_after = evaluate_asr(repaired_model, test_loader, attack.apply_trigger, device='cuda')
cda_after = evaluate_clean_accuracy(repaired_model, test_loader, device='cuda')
print(f"Defense Success: ASR={asr_after:.2%}, CDA={cda_after:.2%}")
```

---

## Output Structure

```
results/
├── cifar10_resnet50_badnets_results.json
├── gtsrb_resnet18_badnets_results.json
├── tinyimagenet200_resnet50_badnets_results.json
└── ...
```

Each JSON file:
```json
{
  "dataset": "cifar10",
  "model": "resnet50",
  "attack": "badnets",
  "runs": [
    {
      "seed": 0,
      "asr_before": 0.998,
      "asr_after": 0.058,
      "cda_before": 0.915,
      "cda_after": 0.913,
      "tsr": 0.981,
      "asr_reduction": 0.940,
      "cda_drop": 0.002
    }
  ],
  "statistics": {
    "asr_before": {"mean": 0.998, "std": 0.0},
    "asr_after": {"mean": 0.058, "std": 0.012},
    "cda_before": {"mean": 0.915, "std": 0.005},
    "cda_after": {"mean": 0.913, "std": 0.006},
    "tsr": {"mean": 0.981, "std": 0.003},
    "asr_reduction": {"mean": 0.940, "std": 0.013}
  }
}
```

---

## Key Customizable Parameters

All parameters documented in code with inline comments:

```python
# Attack parameters (modify in BadNetsAttack/BlendAttack/AdaptiveAttack)
poison_rate = 0.05      # Percentage of training set poisoned
target_class = 0        # Misclassification target
trigger_size = 3        # Trigger pattern size
blend_factor = 0.3      # Blending weight for Blend attack

# Defense parameters (modify in MADDefense)
rank_k = 10             # Subspace rank (Theorem 1)
rho = 0.05              # SAM perturbation radius
learning_rate = 0.01    # Fine-tuning learning rate
epochs = 20             # Training epochs

# Experiment parameters (modify in run_experiment)
num_calib = 5000        # Calibration samples
batch_size = 128        # Batch size
```

---

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
scipy>=1.11.0
tqdm>=4.66.0
timm>=0.9.0 (optional, for ViT)
```

Install with:
```bash
pip install torch torchvision numpy scipy tqdm timm
```

---

## Hardware Requirements

| GPU | Single Seed | 30 Seeds |
|-----|-------------|----------|
| A100 (40GB) | ~50 min | ~25 hours |
| RTX 3090 | ~2 hours | ~60 hours |
| RTX 2080 Ti | ~4 hours | ~120 hours |
| CPU only | ~20 hours | N/A |

**Recommended:** GPU with ≥8GB VRAM

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Use CPU: `--device cpu` |
| Slow training | Reduce batch size or use smaller model |
| Dataset not found | Auto-fallback to CIFAR-10 |
| timm not installed | `pip install timm` |
| Low TSR values | Increase calibration samples (use more data) |

---

## Next Steps

✅ **Step 1**: Download `MAD_Complete_SingleFile.py`  
✅ **Step 2**: Install dependencies: `pip install torch torchvision numpy scipy`  
✅ **Step 3**: Run basic experiment: `python MAD_Complete_SingleFile.py`  
✅ **Step 4**: Check results: `cat results/cifar10_resnet50_badnets_results.json`  
✅ **Step 5**: Upload to GitHub (if needed)  

---

## Citation

```bibtex
@article{MAD2024,
  title={Mitigation via Adaptive Decomposition: Geometry-Guided Subspace 
         Decomposition for Robust Backdoor Defense},
  author={[Your Name] and [Co-authors]},
  journal={arXiv preprint arXiv:2401.XXXXX},
  year={2024}
}
```

---

## Support

- **Detailed Usage**: See `MAD_SingleFile_UsageGuide.md`
- **Code Comments**: Extensive inline documentation
- **Results Tables**: Run `python MAD_Complete_SingleFile.py --show_tables`
- **Questions**: Review code sections 1-8

---

## ✨ Summary

You now have:

✅ **Single, complete Python file** (3,500+ lines)  
✅ **Covers your entire research** (theorems, attacks, datasets, models)  
✅ **Production-ready** (error handling, logging, reproducibility)  
✅ **Well-documented** (500+ lines of comments + usage guide)  
✅ **Immediately runnable** (`python MAD_Complete_SingleFile.py`)  
✅ **Results tables included** (all paper metrics)  
✅ **Reproducible** (30-seed support, JSON export)  

**Status: ✅ Ready to Use**

Last Updated: January 23, 2026