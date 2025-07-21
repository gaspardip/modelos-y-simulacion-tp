#!/usr/bin/env python3
"""
Simple pure Python test to verify time-weighted averaging logic.
"""

def time_weighted_average_python(r_values, dt_sample):
    """
    Pure Python version of time_weighted_average for testing.
    """
    if len(r_values) < 2:
        return r_values[0] if len(r_values) == 1 else 0.0
    
    # Trapezoidal integration: ∫r(t)dt ≈ dt * [r0/2 + r1 + r2 + ... + rN/2]
    integral = 0.5 * (r_values[0] + r_values[-1]) + sum(r_values[1:-1])
    total_time = (len(r_values) - 1) * dt_sample
    
    # Return time-averaged value: (1/T) ∫ r(t) dt
    return integral * dt_sample / total_time

def main():
    print("=== Testing Time-Weighted Order Parameter Averaging ===")
    
    # Test 1: Constant values should give same result as arithmetic mean
    print("Test 1: Constant values")
    r_values = [0.5, 0.5, 0.5, 0.5]
    dt_sample = 0.5
    result = time_weighted_average_python(r_values, dt_sample)
    expected = 0.5
    print(f"  Result: {result:.6f} (expected: {expected:.6f})")
    assert abs(result - expected) < 1e-6, f"Expected {expected}, got {result}"
    print("  ✓ PASSED")
    
    # Test 2: Linear ramp from 0 to 1
    print("\nTest 2: Linear ramp 0→1")
    r_values = [0.0, 0.33, 0.67, 1.0]
    result = time_weighted_average_python(r_values, dt_sample)
    expected = 0.5  # Integral of linear ramp from 0 to 1 over [0,1] is 0.5
    print(f"  Result: {result:.6f} (expected: {expected:.6f})")
    assert abs(result - expected) < 0.02, f"Expected {expected}, got {result}"
    print("  ✓ PASSED")
    
    # Test 3: Single value
    print("\nTest 3: Single value")
    r_values = [0.7]
    result = time_weighted_average_python(r_values, dt_sample)
    expected = 0.7
    print(f"  Result: {result:.6f} (expected: {expected:.6f})")
    assert abs(result - expected) < 1e-6, f"Expected {expected}, got {result}"
    print("  ✓ PASSED")
    
    # Test 4: Compare with arithmetic mean
    print("\nTest 4: Comparison with arithmetic mean")
    r_values = [0.3, 0.4, 0.6, 0.8, 0.7, 0.5, 0.4, 0.3, 0.5, 0.6]
    dt_sample = 0.5
    
    # Old method (simple arithmetic mean)
    arithmetic_mean = sum(r_values) / len(r_values)
    
    # New method (time-weighted)
    time_weighted = time_weighted_average_python(r_values, dt_sample)
    
    print(f"  Arithmetic mean: {arithmetic_mean:.6f}")
    print(f"  Time-weighted:   {time_weighted:.6f}")
    print(f"  Difference:      {abs(time_weighted - arithmetic_mean):.6f}")
    
    # They should be close but not necessarily identical
    assert abs(time_weighted - arithmetic_mean) < 0.1, "Methods differ too much"
    print("  ✓ PASSED")
    
    # Test 5: Verify simulation parameters
    print("\nTest 5: Simulation parameters")
    T_MEASURE = 5.0
    DT = 0.05
    
    expected_steps = int(T_MEASURE / DT)  # Should be 100
    expected_samples = expected_steps // 10  # Should be 10
    sampling_interval = 10 * DT  # Should be 0.5
    
    print(f"  Measurement steps: {expected_steps}")
    print(f"  Expected samples: {expected_samples}")
    print(f"  Sampling interval: {sampling_interval} time units")
    
    assert expected_steps == 100, f"Expected 100 steps, got {expected_steps}"
    assert expected_samples == 10, f"Expected 10 samples, got {expected_samples}"
    assert abs(sampling_interval - 0.5) < 1e-10, f"Expected 0.5, got {sampling_interval}"
    print("  ✓ PASSED")
    
    print("\n✅ All tests passed! Time-weighted averaging implementation logic is correct.")
    print("Note: Full integration test with CuPy requires GPU environment.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        exit(1)