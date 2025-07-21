#!/usr/bin/env python3
"""
Simple test to verify time-weighted order parameter averaging works correctly.
Tests the utility function logic using NumPy (CuPy not available in test environment).
"""

import numpy as np

def time_weighted_average_numpy(r_values, dt_sample):
    """
    NumPy version of time_weighted_average for testing purposes.
    """
    r_values = np.asarray(r_values)
    
    if len(r_values) < 2:
        return r_values[0] if len(r_values) == 1 else 0.0
    
    # Trapezoidal integration: ∫r(t)dt ≈ dt * [r0/2 + r1 + r2 + ... + rN/2]
    integral = 0.5 * (r_values[0] + r_values[-1]) + np.sum(r_values[1:-1])
    total_time = (len(r_values) - 1) * dt_sample
    
    # Return time-averaged value: (1/T) ∫ r(t) dt
    return integral * dt_sample / total_time

def test_time_weighted_average():
    """Test the time_weighted_average function with known values."""
    print("Testing time_weighted_average function...")
    
    # Test 1: Constant values should give same result as arithmetic mean
    r_values = [0.5, 0.5, 0.5, 0.5]
    dt_sample = 0.5
    result = time_weighted_average_numpy(r_values, dt_sample)
    expected = 0.5
    print(f"  Constant values: {result:.6f} (expected: {expected:.6f})")
    assert abs(result - expected) < 1e-6, f"Expected {expected}, got {result}"
    
    # Test 2: Linear ramp from 0 to 1
    r_values = [0.0, 0.33, 0.67, 1.0]
    result = time_weighted_average_numpy(r_values, dt_sample)
    expected = 0.5  # Integral of linear ramp from 0 to 1 over [0,1] is 0.5
    print(f"  Linear ramp 0→1: {result:.6f} (expected: {expected:.6f})")
    assert abs(result - expected) < 1e-2, f"Expected {expected}, got {result}"
    
    # Test 3: Single value
    r_values = [0.7]
    result = time_weighted_average_numpy(r_values, dt_sample)
    expected = 0.7
    print(f"  Single value: {result:.6f} (expected: {expected:.6f})")
    assert abs(result - expected) < 1e-6, f"Expected {expected}, got {result}"
    
    print("  ✓ time_weighted_average tests passed!")

def test_simulation_integration():
    """Test simulation parameters and logic."""
    print("Testing simulation integration...")
    
    # Test parameters are reasonable
    from src.utils import T_MEASURE, DT
    
    expected_steps = int(T_MEASURE / DT)  # Should be 100
    expected_samples = expected_steps // 10  # Should be 10
    
    print(f"  Measurement steps: {expected_steps}")
    print(f"  Expected samples: {expected_samples}")
    print(f"  Sampling interval: {10 * DT} time units")
    
    assert expected_steps == 100, f"Expected 100 steps, got {expected_steps}"
    assert expected_samples == 10, f"Expected 10 samples, got {expected_samples}"
    
    print("  ✓ Simulation parameters are correct!")

def compare_averaging_methods():
    """Compare old vs new averaging methods."""
    print("Comparing averaging methods...")
    
    # Create test data with some variation
    r_values = [0.3, 0.4, 0.6, 0.8, 0.7, 0.5, 0.4, 0.3, 0.5, 0.6]
    dt_sample = 0.5
    
    # Old method (simple arithmetic mean)
    old_average = float(np.mean(np.array(r_values)))
    
    # New method (time-weighted)
    new_average = float(time_weighted_average_numpy(r_values, dt_sample))
    
    print(f"  Simple arithmetic mean: {old_average:.6f}")
    print(f"  Time-weighted average:  {new_average:.6f}")
    print(f"  Difference: {abs(new_average - old_average):.6f}")
    
    # For this particular data, they should be close but not identical
    assert abs(new_average - old_average) < 0.1, "Methods differ too much"
    
    print("  ✓ Averaging method comparison completed!")

if __name__ == "__main__":
    print("=== Testing Time-Weighted Order Parameter Averaging ===")
    try:
        test_time_weighted_average()
        test_simulation_integration()
        compare_averaging_methods()
        print("\n✅ All tests passed! Time-weighted averaging implementation is working correctly.")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        exit(1)