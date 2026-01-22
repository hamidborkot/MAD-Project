"""
Utility functions for data loading, model handling, etc.
"""

import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import random


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_cifar_loaders(batch_size=64, train_subset=None, test_subset=None, 
                     num_workers=4, seed=42):
    """Load CIFAR-10 datasets"""
    set_seed(seed)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                           (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                           (0.2023, 0.1994, 0.2010)),
    ])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)

    if train_subset is not None:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(train_set), size=min(train_subset, len(train_set)),
                            replace=False)
        train_set = Subset(train_set, indices.tolist())

    if test_subset is not None:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(test_set), size=min(test_subset, len(test_set)),
                            replace=False)
        test_set = Subset(test_set, indices.tolist())

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)

    return train_loader, test_loader


def create_model(model_name='resnet18', num_classes=10, pretrained=True):
    """Create model from name"""
    if model_name == 'resnet18':
        model = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == 'vgg19':
        model = torchvision.models.vgg19(
            weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None
        )
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    elif model_name == 'densenet121':
        model = torchvision.models.densenet121(
            weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        )
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


def flatten_parameters(model):
    """Flatten all model parameters into single vector"""
    return torch.cat([p.view(-1) for p in model.parameters()])


def extract_features(model, images, layer_name=None):
    """Extract features from intermediate layer"""
    # Hook to capture intermediate activations
    features = {}

    def get_hook(name):
        def hook(model, input, output):
            features[name] = output.detach()
        return hook

    if layer_name:
        layer = dict(model.named_modules())[layer_name]
        layer.register_forward_hook(get_hook(layer_name))

    with torch.no_grad():
        _ = model(images)

    return features.get(layer_name, None) if layer_name else None
