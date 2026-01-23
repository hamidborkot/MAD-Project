#!/usr/bin/env python3
"""
TEST SUITE: Test Infrastructure Setup
Utilities for test execution, reporting, and validation
"""

import unittest
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

class TestConfig:
    """Global test configuration"""
    VERBOSE = True
    SAVE_RESULTS = True
    RESULTS_DIR = Path('./test_results')
    DEVICE = 'cuda'
    SEED = 42

class TestResult:
    """Stores test results"""
    def __init__(self, test_name: str, status: str, time: float, message: str = ""):
        self.test_name = test_name
        self.status = status  # PASS, FAIL, ERROR
        self.time = time
        self.message = message
        self.timestamp = datetime.now().isoformat()

class TestReporter:
    """Reports test results"""
    def __init__(self):
        self.results: List[TestResult] = []
        TestConfig.RESULTS_DIR.mkdir(exist_ok=True)
    
    def add_result(self, result: TestResult):
        self.results.append(result)
    
    def save_json(self, filename: str):
        """Save results to JSON"""
        output_path = TestConfig.RESULTS_DIR / filename
        data = [
            {
                'test_name': r.test_name,
                'status': r.status,
                'time': r.time,
                'message': r.message,
                'timestamp': r.timestamp
            }
            for r in self.results
        ]
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✓ Results saved to {output_path}")
    
    def print_summary(self):
        """Print test summary"""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.status == 'PASS')
        failed = sum(1 for r in self.results if r.status == 'FAIL')
        errors = sum(1 for r in self.results if r.status == 'ERROR')
        
        print(f"\n{'='*60}")
        print(f"TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total:  {total}")
        print(f"Passed: {passed} ✓")
        print(f"Failed: {failed} ✗")
        print(f"Errors: {errors} ⚠")
        print(f"Success Rate: {100*passed/total:.1f}%")
        print(f"{'='*60}\n")

# Global reporter
REPORTER = TestReporter()
