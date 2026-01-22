"""
Defense evaluation metrics
"""

import numpy as np
import torch
import torch.nn.functional as F


class DefenseMetrics:
    """Compute defense evaluation metrics"""

    @staticmethod
    def clean_accuracy(model, clean_loader, device='cuda'):
        """Clean accuracy on benign data"""
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for images, labels in clean_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return correct / total if total > 0 else 0.0

    @staticmethod
    def attack_success_rate(model, poisoned_loader, target_label=0, device='cuda'):
        """Attack success rate (ASR) on poisoned data"""
        model.eval()
        success, total = 0, 0

        with torch.no_grad():
            for images, _ in poisoned_loader:
                images = images.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                success += (preds == target_label).sum().item()
                total += images.shape[0]

        return success / total if total > 0 else 0.0

    @staticmethod
    def robust_accuracy(model, poisoned_loader, true_labels, device='cuda'):
        """Accuracy preserving true labels on poisoned data"""
        model.eval()
        correct, total = 0, 0

        idx = 0
        with torch.no_grad():
            for images, _ in poisoned_loader:
                images = images.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)

                batch_size = images.shape[0]
                labels = torch.tensor(true_labels[idx:idx+batch_size], 
                                    device=device, dtype=torch.long)
                correct += (preds == labels).sum().item()
                total += batch_size
                idx += batch_size

        return correct / total if total > 0 else 0.0

    @staticmethod
    def TSR_score(features_before, features_after):
        """Total Structure Retention"""
        if isinstance(features_before, torch.Tensor):
            fb = features_before.cpu().detach().numpy()
        else:
            fb = features_before

        if isinstance(features_after, torch.Tensor):
            fa = features_after.cpu().detach().numpy()
        else:
            fa = features_after

        var_before = np.var(fb, axis=0).mean()
        var_after = np.var(fa, axis=0).mean()

        if var_before < 1e-10:
            return 0.0

        TSR = abs(np.log10(var_before + 1e-12) - np.log10(var_after + 1e-12))
        return TSR

    @staticmethod
    def computational_overhead(time_with_defense, time_without_defense):
        """Compute overhead percentage"""
        if time_without_defense == 0:
            return 0.0
        overhead_pct = ((time_with_defense - time_without_defense) / 
                       time_without_defense) * 100
        return max(0.0, overhead_pct)
