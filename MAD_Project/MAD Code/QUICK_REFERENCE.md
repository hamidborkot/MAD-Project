# MAD Defense - Quick Reference

## Installation

```bash
pip install -r requirements.txt
```

## Basic Usage

```python
from mad_defense import MADMitigator
from utils import get_cifar_loaders, create_model

# 1. Load data and model
train_loader, test_loader = get_cifar_loaders()
model = create_model('resnet18')

# 2. Create MAD mitigator
mitigator = MADMitigator(model, rank_K=10, device='cuda')

# 3. Identify adversarial subspace
mitigator.identify_subspace(train_loader, num_samples=500)

# 4. Apply localized SAM repair
mitigator.repair(train_loader, epochs=5, lr=0.001)

# 5. Evaluate
from metrics import DefenseMetrics
accuracy = DefenseMetrics.clean_accuracy(model, test_loader)
```

## Key Components

### 1. MADDefense Class
- `compute_eigensubspace()`: Identify adversarial subspace U_B
- `project_remove_subspace()`: Apply projection P = I - U U^T
- `compute_TSR()`: Measure structure retention

### 2. Localized SAM
- Perturbation in adversarial subspace only
- Robustness against fine-tuned attacks
- Minimal clean accuracy drop

### 3. MADMitigator Pipeline
- Complete end-to-end defense pipeline
- Subspace identification + localized SAM + projection

## Correction Summary

✓ FIXED: Projection operator (now correctly removes subspace)
✓ FIXED: Eigendecomposition (uses G^T G for proper covariance)
✓ FIXED: Localized SAM integration (proper perturbation step)
✓ FIXED: TSR metric (variance-based structure retention)
✓ ADDED: VGG-19 support (proper feature extraction)
✓ ADDED: Error handling (graceful fallback for edge cases)
✓ ADDED: Complete metrics (CA, ASR, CDA, TSR)

## Configuration

```python
config = {
    'rank_K': 10,           # Adversarial subspace dimension
    'mad_epochs': 5,        # SAM repair epochs
    'rho': 0.05,           # SAM perturbation radius
    'device': 'cuda',      # GPU device
    'batch_size': 64,      # Batch size
    'num_samples': 500     # Gradients to collect
}
```

## Supported Models

- ResNet-18/50/152
- VGG-19 (now fully integrated!)
- DenseNet-121
- Easy to extend with custom architectures

## Citation

If you use MAD, please cite:

```
@paper{MAD2024,
    title={Manifold-Adaptive Defense: Defending Against Feature Collision Attacks},
    year={2024}
}
```
