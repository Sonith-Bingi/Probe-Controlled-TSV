#!/usr/bin/env python
"""
Quick test of the TSV comparison pipeline

This script runs a minimal test to verify all components work together.
"""

import os
import sys
import torch
import logging
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "experiments" / "probe_controlled_tsv"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all required modules can be imported."""
    logger.info("Testing imports...")
    
    try:
        from experiments.probe_controlled_tsv.train_tsv_original import train_original_tsv, load_tqa_data_for_tsv
        from experiments.probe_controlled_tsv.run_full_experiment import load_tsv, load_tsv_original, run_experiment
        from experiments.probe_controlled_tsv.models.probe import MLPProbe, LinearProbe
        logger.info("✅ All imports successful")
        return True
    except Exception as e:
        logger.error(f"❌ Import failed: {e}")
        return False


def test_file_structure():
    """Test that required files exist.""" 
    logger.info("Testing file structure...")
    
    required_files = [
        "experiments/probe_controlled_tsv/train_tsv_original.py",
        "experiments/probe_controlled_tsv/run_full_experiment.py", 
        "experiments/probe_controlled_tsv/models/probe.py",
        "train_utils.py",
        "sinkhorn_knopp.py"
    ]
    
    missing = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing.append(file_path)
    
    if missing:
        logger.error(f"❌ Missing files: {missing}")
        return False
    
    logger.info("✅ All required files present")
    return True


def test_cuda():
    """Test CUDA availability."""
    logger.info("Testing CUDA...")
    
    if torch.cuda.is_available():
        logger.info(f"✅ CUDA available: {torch.cuda.get_device_name()}")
        logger.info(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return True
    else:
        logger.warning("⚠️  CUDA not available, will use CPU (slower)")
        return True


def main():
    """Run all tests."""
    print("=" * 50)
    print("TSV COMPARISON PIPELINE - QUICK TEST")
    print("=" * 50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Imports", test_imports), 
        ("CUDA", test_cuda)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\nTesting {name}...")
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            logger.error(f"❌ {name} test crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)
    
    all_passed = True
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{name:<20} {status}")
        if not success:
            all_passed = False
    
    print("=" * 50)
    
    if all_passed:
        print("🎉 All tests passed! Pipeline is ready to run.")
        print("\nTo run the full comparison:")
        print("  python run_comparison_pipeline.py --quick")
        print("\nTo run with more samples:")
        print("  python run_comparison_pipeline.py")
    else:
        print("❌ Some tests failed. Please fix issues before running pipeline.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())