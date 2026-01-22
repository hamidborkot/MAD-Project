# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-01-23

### Added
- Initial release of MAD (Mitigation via Adaptive Decomposition)
- Core defense implementation with gradient covariance eigenanalysis
- Localized sharpness-aware optimization
- Support for 5 backdoor attacks: BadNets, Blend, WaNet, GOA, LASE
- Support for 5 architectures: ResNet-18/50, VGG-19, DenseNet-121, MobileNet-V2, ViT-Tiny
- CIFAR-10, GTSRB, ImageNet-1K experimental pipelines
- Calibration density scaling experiments
- Distributed trigger failure mode analysis
- Comprehensive ablation studies
- Theoretical analysis (Theorems 1-3, Lemmas, Corollaries)
- TSR (Total Structure Retention) metric
- Command-line interface for easy usage
- Extensive documentation and tutorials
- Pretrained backdoored models
- Reproducibility scripts

### Baselines Included
- Fine-Tuning (FT)
- Fine-Pruning (FP)
- Adversarial Neuron Pruning (ANP)
- FT-SAM
- BFA-Det
- FBBD

### Results
- ASR ≤ 5.12% across all attacks (ResNet-18 CIFAR-10)
- CDA within 0.4 pp of baseline
- Statistical significance: p ≤ 1.05×10⁻⁶
- TSR ≈ 0.98 at K=10
- 60% faster convergence than FT-SAM

### Documentation
- Comprehensive README with examples
- API documentation
- Contributing guidelines
- License (MIT)
- Installation instructions
- Troubleshooting guide

---

## [Unreleased]

### Planned
- Multi-layer subspace decomposition
- Hierarchical defense for distributed backdoors
- Support for Transformer architectures (BERT, GPT)
- Real-time deployment optimization
- Distributed training support
- Hyperparameter auto-tuning
- Web-based visualization dashboard
- Model zoo with more architectures
