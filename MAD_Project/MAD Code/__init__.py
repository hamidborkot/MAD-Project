"""
Manifold-Adaptive Defense (MAD)
Corrected implementation with proper eigendecomposition and projection
"""

from .mad_defense import MADDefense, MADMitigator, localized_SAM_step
from .attacks import BadNetsAttack, FeatureCollisionAttack, AdaptiveAttack
from .metrics import DefenseMetrics
from .utils import set_seed, get_cifar_loaders, create_model

__version__ = "1.0.0"
__author__ = "MAD Team"

__all__ = [
    'MADDefense',
    'MADMitigator',
    'localized_SAM_step',
    'BadNetsAttack',
    'FeatureCollisionAttack',
    'AdaptiveAttack',
    'DefenseMetrics',
    'set_seed',
    'get_cifar_loaders',
    'create_model',
]
