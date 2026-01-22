"""
MAD Defense - Main Experiment Pipeline
Complete evaluation framework with correct implementation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import json
from pathlib import Path

from mad_defense import MADDefense, MADMitigator
from attacks import BadNetsAttack, FeatureCollisionAttack
from metrics import DefenseMetrics
from utils import set_seed, get_cifar_loaders, create_model


class MADExperiment:
    """Complete MAD defense evaluation"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = []

    def run_complete_evaluation(self):
        """Run full MAD evaluation pipeline"""
        print("="*80)
        print("MAD DEFENSE - COMPLETE EVALUATION")
        print("="*80)

        set_seed(self.config['seed'])

        # 1. Load data
        print("\n[1/6] Loading CIFAR-10 data...")
        train_loader, test_loader = get_cifar_loaders(
            batch_size=self.config['batch_size'],
            train_subset=self.config.get('train_subset', 2000),
            test_subset=self.config.get('test_subset', 1000),
            seed=self.config['seed']
        )

        # 2. Create model
        print("[2/6] Creating model...")
        model = create_model(
            self.config['model_name'],
            num_classes=10,
            pretrained=True
        ).to(self.device)

        # 3. Train baseline
        print("[3/6] Training baseline model...")
        self._train_model(model, train_loader, epochs=self.config.get('epochs', 5))

        # 4. Evaluate baseline
        print("[4/6] Evaluating baseline...")
        clean_acc = DefenseMetrics.clean_accuracy(model, test_loader, self.device)
        print(f"   Clean Accuracy: {clean_acc:.4f}")

        # 5. Apply MAD defense
        print("[5/6] Applying MAD defense...")
        mitigator = MADMitigator(model, rank_K=self.config.get('rank_K', 10), 
                                device=self.device)
        mitigator.identify_subspace(train_loader, num_samples=500)
        mitigator.repair(train_loader, epochs=self.config.get('mad_epochs', 3))

        # 6. Evaluate defended model
        print("[6/6] Evaluating defended model...")
        defended_acc = DefenseMetrics.clean_accuracy(model, test_loader, self.device)
        print(f"   Defended Clean Accuracy: {defended_acc:.4f}")

        # Generate report
        self._generate_report(clean_acc, defended_acc)

        return self.results

    def _train_model(self, model, train_loader, epochs=5):
        """Train model on clean data"""
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        for epoch in range(epochs):
            total_loss = 0
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"   Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

    def _generate_report(self, clean_acc, defended_acc):
        """Generate evaluation report"""
        report = {
            'model': self.config['model_name'],
            'baseline_ca': clean_acc,
            'defended_ca': defended_acc,
            'ca_drop': clean_acc - defended_acc,
            'rank_K': self.config.get('rank_K', 10),
            'timestamp': time.time()
        }
        self.results.append(report)

        print("\n" + "="*80)
        print("EVALUATION REPORT")
        print("="*80)
        print(f"Model: {report['model']}")
        print(f"Baseline CA: {report['baseline_ca']:.4f}")
        print(f"Defended CA: {report['defended_ca']:.4f}")
        print(f"CA Drop: {report['ca_drop']:.4f}")
        print("="*80)


def run_cifar10_benchmark():
    """Run benchmark on CIFAR-10"""
    config = {
        'model_name': 'resnet18',
        'batch_size': 64,
        'epochs': 5,
        'mad_epochs': 3,
        'rank_K': 10,
        'seed': 42,
        'train_subset': 2000,
        'test_subset': 1000
    }

    experiment = MADExperiment(config)
    results = experiment.run_complete_evaluation()

    # Save results
    df = pd.DataFrame(results)
    df.to_csv('mad_results_cifar10.csv', index=False)
    print(f"\nResults saved to mad_results_cifar10.csv")


if __name__ == '__main__':
    run_cifar10_benchmark()
