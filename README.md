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
git clone https://github.com/yourusername/MAD.git
cd MAD

# Create virtual environment
python -m venv mad_env
source mad_env/bin/activate  # On Windows: mad_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in editable mode
pip install -e .
```

### Dependencies
```txt
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
tqdm>=4.65.0
tensorboard>=2.13.0
scikit-learn>=1.3.0
pandas>=2.0.0
```

---

## 🚀 Quick Start

### Basic Usage

```python
from mad import MADDefense
import torch
from torchvision import models

# Load potentially backdoored model
model = models.resnet18(pretrained=False)
model.load_state_dict(torch.load('backdoored_model.pth'))

# Initialize MAD defense
defense = MADDefense(
    model=model,
    calibration_data=clean_dataset,  # Small trusted dataset (1000 samples)
    K=10,                             # Subspace dimension
    rho=0.05,                         # SAM perturbation radius
    num_epochs=20,                    # Training epochs
    lr=0.01                           # Learning rate
)

# Run defense
repaired_model = defense.defend()

# Evaluate
asr, cda = defense.evaluate(test_loader, trigger_loader)
print(f"Attack Success Rate: {asr:.2f}%")
print(f"Clean Data Accuracy: {cda:.2f}%")
```

### Command Line Interface

```bash
# Train and defend against BadNets on CIFAR-10
python main.py \
    --dataset cifar10 \
    --model resnet18 \
    --attack badnets \
    --poison-rate 0.05 \
    --calib-size 1000 \
    --defense mad \
    --K 10 \
    --rho 0.05 \
    --epochs 20

# Reproduce GTSRB experiments with 5-fold CV
python experiments/gtsrb_eval.py \
    --model resnet18 \
    --cv-folds 5 \
    --seeds 30

# Run calibration density scaling experiments
python experiments/calibration_scaling.py \
    --model resnet50 \
    --calib-sizes 500,1000,2500,5000,10000
```

---

## 📁 Repository Structure

```
MAD/
├── mad/
│   ├── __init__.py
│   ├── defense.py              # Main MAD defense implementation
│   ├── subspace.py             # Gradient covariance eigenanalysis
│   ├── sam_localized.py        # Localized SAM optimizer
│   ├── metrics.py              # TSR, ASR, CDA calculations
│   └── utils.py                # Helper functions
├── attacks/
│   ├── badnets.py              # BadNets implementation
│   ├── blend.py                # Blend attack
│   ├── wanet.py                # WaNet attack
│   ├── goa.py                  # GOA (adaptive)
│   └── lase.py                 # LASE (adaptive)
├── baselines/
│   ├── fine_tuning.py          # Standard fine-tuning
│   ├── fine_pruning.py         # Fine-Pruning baseline
│   ├── anp.py                  # Adversarial Neuron Pruning
│   ├── ft_sam.py               # FT-SAM baseline
│   ├── bfa_det.py              # BFA-Det baseline
│   └── fbbd.py                 # FBBD baseline
├── experiments/
│   ├── cifar10_main.py         # CIFAR-10 experiments
│   ├── gtsrb_eval.py           # GTSRB 5-fold CV
│   ├── imagenet_eval.py        # ImageNet-1K experiments
│   ├── calibration_scaling.py  # Theorem 2 validation
│   ├── distributed_triggers.py # Failure mode analysis
│   └── ablation_studies.py     # Ablation experiments
├── models/
│   ├── resnet.py               # ResNet architectures
│   ├── vgg.py                  # VGG with BatchNorm
│   ├── densenet.py             # DenseNet-121
│   ├── mobilenet.py            # MobileNet-V2
│   └── vit.py                  # Vision Transformer
├── data/
│   ├── cifar10_loader.py       # CIFAR-10 dataloader
│   ├── gtsrb_loader.py         # GTSRB dataloader
│   └── imagenet_loader.py      # ImageNet subset loader
├── scripts/
│   ├── train_poisoned.sh       # Train backdoored models
│   ├── run_all_experiments.sh  # Reproduce all results
│   └── plot_results.py         # Generate figures
├── configs/
│   ├── cifar10.yaml            # CIFAR-10 configuration
│   ├── gtsrb.yaml              # GTSRB configuration
│   └── imagenet.yaml           # ImageNet configuration
├── tests/
│   ├── test_defense.py         # Unit tests
│   ├── test_subspace.py        # Subspace identification tests
│   └── test_metrics.py         # Metric calculation tests
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup
├── README.md                   # This file
└── LICENSE                     # MIT License
```

---

## 📊 Experimental Results

### Main Results on CIFAR-10

| Method | CDA (↑) | BadNets ASR (↓) | Blend ASR (↓) | WaNet ASR (↓) | GOA ASR (↓) | LASE ASR (↓) |
|--------|---------|-----------------|---------------|---------------|-------------|--------------|
| FT | 90.02 | 71.30 | 75.12 | 68.45 | - | - |
| FP | 89.55 | 35.10 | 38.05 | 42.10 | - | - |
| ANP | 90.15 | 25.50 | 29.33 | 10.45 | - | - |
| FT-SAM | 90.35 | 14.28 | 17.15 | 9.81 | 18.45 | 15.60 |
| BFA-Det | 90.18 | 15.60 | 14.28 | 10.45 | 15.60 | 13.45 |
| FBBD | 90.22 | 14.28 | 13.15 | 9.15 | 14.28 | 12.45 |
| **MAD (Ours)** | **91.10** | **2.12** | **1.85** | **3.02** | **3.85** | **2.92** |

*ResNet-18 on CIFAR-10. Mean over 30 seeds. All MAD vs. FT-SAM: p ≤ 1.05×10⁻⁶*

### Cross-Architecture Generalization

| Model | Params | Calib. Size | Baseline ASR | MAD ASR | CDA | CDA Drop |
|-------|--------|-------------|--------------|---------|-----|----------|
| ResNet-18 | 11.2M | 1K | 99.8% | 5.8% | 91.1% | 0.11 pp |
| DenseNet-121 | 7.0M | 1K | 98.5% | 2.2% | 91.1% | 0.12 pp |
| MobileNet-V2 | 2.3M | 1K | 97.8% | 3.1% | 90.9% | 0.28 pp |
| ViT-Tiny | 5.7M | 1K | 94.1% | 3.9% | 90.9% | 0.45 pp |
| ResNet-50 | 23.5M | 5K | 98.1% | 4.9% | 91.5% | 0.18 pp |

### Safety-Critical Domain (GTSRB)

| Model | MAD ASR | CDA | Std Dev |
|-------|---------|-----|---------|
| ResNet-18 | 1.89% | 97.41% | ±0.14 |
| VGG-19 | 2.50% | 96.04% | ±0.08 |

*5-fold cross-validation on German Traffic Sign Recognition Benchmark*

### Calibration Density Validation (Theorem 2)

| Calib. Size | ρ (×10⁻⁴) | TSR | MAD ASR | CDA |
|-------------|-----------|-----|---------|-----|
| 500 | 0.024 | 0.62 | 42.5% | 87.2% |
| 1,000 | 0.049 | 0.78 | 28.3% | 88.8% |
| 2,500 | 0.122 | 0.89 | 18.3% | 88.1% |
| **5,000** | **0.244** | **0.97** | **4.9%** | **91.5%** |

*ResNet-50 on CIFAR-10. Threshold: ρ ≥ 0.21×10⁻⁴*

---

## 📚 Datasets & Models

### Supported Datasets
- **CIFAR-10**: 32×32 RGB, 10 classes, 50K train / 10K test
- **GTSRB**: Traffic signs, 43 classes, variable resolution
- **ImageNet-1K**: 224×224 RGB, 50-class subset

### Supported Architectures
- ResNet-18, ResNet-50
- VGG-19 (with BatchNorm)
- DenseNet-121
- MobileNet-V2
- ViT-Tiny (Vision Transformer)

### Supported Attacks

#### Standard Attacks
- **BadNets**: Localized 3×3 checkerboard trigger
- **Blend**: Semi-transparent overlay (α=0.2)
- **WaNet**: Imperceptible warping field

#### Adaptive Attacks
- **GOA (Gradient-Orthogonal Attack)**: Flattens global loss landscape
- **LASE (Latent Subspace Evasion)**: Targets latent representations

---

## 🎓 Theoretical Guarantees

### Theorem 1: Low-Rank Backdoor Concentration
Backdoor perturbations concentrate in K-dimensional subspace with:
```
‖P_B(θ_bd - θ_clean)‖₂ / ‖θ_bd - θ_clean‖₂ ≥ 1 - O(K log(d/ε) / n_clean)
```

### Theorem 2: Calibration Density Threshold
Subspace alignment error bounded by:
```
E[sin²Θ(B, B̂_K)] ≤ O(Kd / n_clean)
```
Requires: `n_clean ≥ 0.21×10⁻⁴ × d` for TSR ≥ 0.98

### Theorem 3: Defense Effectiveness Bound
Attack success rate decays exponentially:
```
ASR(θ*) ≤ ASR(θ₀) · exp(-ρ²T λ_min(H_B) / 4β) + O(K log d / n_clean)
```

**Full proofs available in Appendix B of the paper.**

---

## 🔬 Reproducing Results

### Step 1: Train Backdoored Models

```bash
# Train models with different attacks
bash scripts/train_poisoned.sh --dataset cifar10 --attack all --seeds 30
```

### Step 2: Run Defense Experiments

```bash
# Main CIFAR-10 experiments (Table 3)
python experiments/cifar10_main.py --config configs/cifar10.yaml

# GTSRB 5-fold CV (Table 4)
python experiments/gtsrb_eval.py --cv-folds 5 --seeds 30

# ImageNet experiments (Table 4)
python experiments/imagenet_eval.py --subset 50class

# Calibration scaling (Table 5, Figure 3)
python experiments/calibration_scaling.py --model resnet50

# Distributed triggers (Table 5)
python experiments/distributed_triggers.py --coverages 5,15,30,50
```

### Step 3: Generate Figures

```bash
# Generate all paper figures
python scripts/plot_results.py --output-dir figures/
```

### Expected Runtime
- CIFAR-10 (ResNet-18): ~2 hours per attack (NVIDIA A100)
- GTSRB (5-fold): ~6 hours total
- ImageNet-1K: ~12 hours (ResNet-50)
- Calibration scaling: ~8 hours

### Pretrained Models
Download pretrained backdoored models:
```bash
bash scripts/download_models.sh
# Models saved to: checkpoints/backdoored/
```

---

## 🎯 Hyperparameter Guidance

### Default Configuration (works for most cases)
```python
config = {
    'K': 10,              # Subspace dimension
    'rho': 0.05,          # SAM perturbation radius
    'epochs': 20,         # Training epochs
    'lr': 0.01,           # Learning rate
    'batch_size': 128,    # Batch size
    'momentum': 0.9,      # SGD momentum
    'weight_decay': 5e-4  # Weight decay
}
```

### Model-Specific Recommendations

| Model | d (final layer dim) | Min Calib Size | Recommended K |
|-------|---------------------|----------------|---------------|
| ResNet-18 | 5,120 | 1,000 | 10 |
| VGG-19 | 4,096 | 1,000 | 10 |
| DenseNet-121 | 1,024 | 500 | 8 |
| ResNet-50 | 20,480 | 5,000 | 10 |
| ViT-Tiny | 192 | 500 | 8 |

**Formula**: `n_clean ≥ 0.21×10⁻⁴ × d`

### Ablation Results
- **K sensitivity**: Performance stable for K ∈ [8, 15]
- **ρ sensitivity**: Optimal range [0.03, 0.07]
- **Early stopping**: Epoch 8-10 achieves 95% final performance

---

## 📈 Performance Metrics

### Total Structure Retention (TSR)
Measures subspace estimation quality:
```python
TSR = ‖P_B Δθ_bd‖₂ / ‖Δθ_bd‖₂
```
- **Target**: TSR ≥ 0.95
- **Typical**: TSR ≈ 0.98 at K=10

### Attack Success Rate (ASR)
Percentage of triggered inputs misclassified to target:
```python
ASR = (# successful triggers) / (# total triggers) × 100%
```

### Clean Data Accuracy (CDA)
Accuracy on benign test set:
```python
CDA = (# correct predictions) / (# total samples) × 100%
```

---

## 🐛 Troubleshooting

### Common Issues

**1. High ASR after defense (>10%)**
- Check calibration size: ensure ρ ≥ 0.21×10⁻⁴
- Verify clean dataset quality: contamination should be <1%
- Try increasing K to 12-15

**2. CDA drop >1 pp**
- Reduce learning rate to 0.005
- Decrease SAM radius ρ to 0.03
- Enable early stopping at epoch 10

**3. Out of memory**
- Reduce batch size to 64 or 32
- Use gradient checkpointing
- For d > 50K, use sparse eigensolvers

**4. Slow eigendecomposition**
- Enable GPU acceleration: `torch.linalg.eigh(C.cuda())`
- Use top-K eigensolver: `torch.lobpcg()`
- Reduce calibration size if ρ already > threshold

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2026 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

[Full license text...]
```

---

## 🙏 Acknowledgments

- Thanks to the PyTorch team for the deep learning framework
- GTSRB dataset providers for traffic sign benchmarks
- Backdoor attack implementations adapted from [BackdoorBox](https://github.com/THUYimingLi/BackdoorBox)
- SAM optimizer from [official implementation](https://github.com/davda54/sam)
- Baseline methods: FT-SAM, ANP, Fine-Pruning, BFA-Det, FBBD

---

## 🔗 Related Work

- **SPECTRE**: Spectral backdoor detection [Liu et al., 2023]
- **FT-SAM**: Fine-tuning with SAM [Chen et al., 2023]
- **ANP**: Adversarial Neuron Pruning [Wu et al., 2021]
- **BFA-Det**: Backdoor-Free Architecture Detection [Wang et al., 2024]
- **FBBD**: Feature-Based Backdoor Detection [Zhang et al., 2024]

---

## 📧 Contact

For questions or issues, please:
- Open an issue on GitHub
- Email: [hamidborkot@stmail.ntu.edu.cn.edu]

---

## 🔄 Version History

- **v1.0.0** (Jan 2026): Initial release
  - Core MAD defense implementation
  - CIFAR-10, GTSRB, ImageNet experiments
  - 5 attack types (BadNets, Blend, WaNet, GOA, LASE)
  - Theoretical analysis and proofs

---

## 🎯 Future Work

- [ ] Multi-layer subspace decomposition
- [ ] Hierarchical defense for distributed backdoors (>30% coverage)
- [ ] Extension to model unlearning and watermark removal
- [ ] Real-time deployment optimization
- [ ] Support for additional architectures (Transformers, EfficientNets)

---

**Star ⭐ this repository if you find it helpful!**
