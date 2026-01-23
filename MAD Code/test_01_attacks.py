#!/usr/bin/env python3
"""
TEST SUITE 1: Backdoor Attacks
Complete test suite for BadNets, Blend, and Adaptive attacks
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
import time

# Import from main implementation
sys.path.insert(0, str(Path(__file__).parent))

class TestBadNetsAttack(unittest.TestCase):
    """Test BadNets attack implementation"""
    
    @classmethod
    def setUpClass(cls):
        """Setup test fixtures"""
        torch.manual_seed(42)
        np.random.seed(42)
        cls.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create dummy model
        cls.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, 10)
        ).to(cls.device)
        
        # Create dummy dataset
        cls.dummy_images = torch.randn(32, 3, 32, 32).to(cls.device)
        cls.dummy_labels = torch.randint(0, 10, (32,)).to(cls.device)
    
    def test_01_trigger_creation(self):
        """Test: BadNets trigger pattern creation"""
        from MAD_Complete_SingleFile import BadNetsAttack
        
        attack = BadNetsAttack(poison_rate=0.05, target_class=0, device=self.device)
        
        # Verify trigger exists
        self.assertIsNotNone(attack.trigger)
        self.assertEqual(attack.trigger.shape[0], 3)  # 3 channels
        self.assertEqual(attack.trigger.shape[1], 3)  # 3x3
        self.assertEqual(attack.trigger.shape[2], 3)
        
        print("✓ Test 1.1: Trigger created successfully")
    
    def test_02_apply_single_trigger(self):
        """Test: Apply trigger to single image"""
        from MAD_Complete_SingleFile import BadNetsAttack
        
        attack = BadNetsAttack(poison_rate=0.05, device=self.device)
        image = self.dummy_images[0:1]  # Single image (1, 3, 32, 32)
        
        triggered = attack.apply_trigger(image)
        
        # Verify shape unchanged
        self.assertEqual(triggered.shape, image.shape)
        # Verify not all zeros
        self.assertFalse(torch.allclose(triggered, image))
        
        print("✓ Test 1.2: Single image trigger applied")
    
    def test_03_apply_batch_trigger(self):
        """Test: Apply trigger to batch of images"""
        from MAD_Complete_SingleFile import BadNetsAttack
        
        attack = BadNetsAttack(poison_rate=0.05, device=self.device)
        batch = self.dummy_images[:8]
        
        triggered = attack.apply_trigger(batch)
        
        # Verify shape
        self.assertEqual(triggered.shape, batch.shape)
        self.assertEqual(triggered.shape[0], 8)
        
        print("✓ Test 1.3: Batch trigger applied")
    
    def test_04_poison_model(self):
        """Test: Poison model with backdoor"""
        from MAD_Complete_SingleFile import BadNetsAttack
        from torch.utils.data import DataLoader, TensorDataset
        
        attack = BadNetsAttack(poison_rate=0.05, target_class=0, device=self.device)
        
        # Create dummy dataloader
        dataset = TensorDataset(self.dummy_images, self.dummy_labels)
        loader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        # Poison
        poisoned_model = attack.poison_model(
            self.model, loader, epochs=2, device=self.device
        )
        
        # Verify model is returned
        self.assertIsNotNone(poisoned_model)
        self.assertEqual(type(poisoned_model), type(self.model))
        
        print("✓ Test 1.4: Model poisoned successfully")


class TestBlendAttack(unittest.TestCase):
    """Test Blend attack implementation"""
    
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(42)
        np.random.seed(42)
        cls.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        cls.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, 10)
        ).to(cls.device)
        
        cls.dummy_images = torch.randn(32, 3, 32, 32).to(cls.device)
        cls.dummy_labels = torch.randint(0, 10, (32,)).to(cls.device)
    
    def test_01_watermark_creation(self):
        """Test: Blend watermark pattern creation"""
        from MAD_Complete_SingleFile import BlendAttack
        
        attack = BlendAttack(poison_rate=0.05, device=self.device)
        
        self.assertIsNotNone(attack.watermark)
        self.assertEqual(attack.watermark.shape[0], 3)
        self.assertEqual(attack.watermark.shape[1], 32)
        
        print("✓ Test 1.5: Watermark created successfully")
    
    def test_02_blend_trigger(self):
        """Test: Apply Blend trigger"""
        from MAD_Complete_SingleFile import BlendAttack
        
        attack = BlendAttack(poison_rate=0.05, blend_factor=0.2, device=self.device)
        image = self.dummy_images[0:1]
        
        triggered = attack.apply_trigger(image)
        
        self.assertEqual(triggered.shape, image.shape)
        self.assertTrue(torch.all(triggered >= 0) and torch.all(triggered <= 1))
        
        print("✓ Test 1.6: Blend trigger applied with clamping")
    
    def test_03_blend_parameters(self):
        """Test: Blend attack with different blend factors"""
        from MAD_Complete_SingleFile import BlendAttack
        
        for blend_factor in [0.1, 0.2, 0.3, 0.5]:
            attack = BlendAttack(poison_rate=0.05, blend_factor=blend_factor, device=self.device)
            self.assertEqual(attack.blend_factor, blend_factor)
        
        print("✓ Test 1.7: Multiple blend factors tested")


class TestAdaptiveAttack(unittest.TestCase):
    """Test Adaptive attacks (GOA, LASE)"""
    
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(42)
        np.random.seed(42)
        cls.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        cls.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, 10)
        ).to(cls.device)
        
        cls.dummy_images = torch.randn(32, 3, 32, 32).to(cls.device)
        cls.dummy_labels = torch.randint(0, 10, (32,)).to(cls.device)
    
    def test_01_goa_trigger(self):
        """Test: GOA (Gradient-Orthogonal) trigger"""
        from MAD_Complete_SingleFile import AdaptiveAttack
        
        attack = AdaptiveAttack(attack_type='goa', device=self.device)
        image = self.dummy_images[0:1]
        
        triggered = attack.apply_trigger(image)
        
        self.assertEqual(triggered.shape, image.shape)
        self.assertTrue(torch.all(triggered >= 0) and torch.all(triggered <= 1))
        
        print("✓ Test 1.8: GOA trigger applied")
    
    def test_02_lase_trigger(self):
        """Test: LASE (Layer-Aware) trigger"""
        from MAD_Complete_SingleFile import AdaptiveAttack
        
        attack = AdaptiveAttack(attack_type='lase', device=self.device)
        image = self.dummy_images[0:1]
        
        triggered = attack.apply_trigger(image)
        
        self.assertEqual(triggered.shape, image.shape)
        self.assertTrue(torch.all(triggered >= 0) and torch.all(triggered <= 1))
        
        print("✓ Test 1.9: LASE trigger applied")
    
    def test_03_adaptive_poison(self):
        """Test: Poison model with adaptive attack"""
        from MAD_Complete_SingleFile import AdaptiveAttack
        from torch.utils.data import DataLoader, TensorDataset
        
        for attack_type in ['goa', 'lase']:
            attack = AdaptiveAttack(attack_type=attack_type, device=self.device)
            dataset = TensorDataset(self.dummy_images, self.dummy_labels)
            loader = DataLoader(dataset, batch_size=8, shuffle=True)
            
            poisoned = attack.poison_model(self.model, loader, epochs=2, device=self.device)
            self.assertIsNotNone(poisoned)
        
        print("✓ Test 1.10: Adaptive attacks tested")


def run_attack_tests():
    """Run all attack tests"""
    print("\n" + "="*60)
    print("RUNNING ATTACK TESTS")
    print("="*60 + "\n")
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestBadNetsAttack))
    suite.addTests(loader.loadTestsFromTestCase(TestBlendAttack))
    suite.addTests(loader.loadTestsFromTestCase(TestAdaptiveAttack))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    result = run_attack_tests()
    sys.exit(0 if result.wasSuccessful() else 1)
