# Phase Normalization Issue Analysis

## Phase Normalization Issue Analysis

### 1. What the Current Code Does

**Location**: `/home/gaspar/src/tp-modelos-y-simulacion/src/utils.py`, lines 71-73

```python
# Normalizar fases al rango [0, 2π] para evitar problemas numéricos
two_pi = thetas.dtype.type(2.0) * cp.pi
cp.remainder(thetas, two_pi, out=thetas)
```

This code is part of the `rk4_step` function (lines 49-75) which performs numerical integration of the Kuramoto model using the 4th-order Runge-Kutta method. After updating the phase angles (`thetas`) in line 69, the code normalizes them to the range [0, 2π) using CuPy's `remainder` function.

### 2. Why cp.remainder Causes Discontinuities

The `cp.remainder` function implements the IEEE remainder operation, which can introduce discontinuities in the Kuramoto model for several reasons:

1. **Abrupt Wrapping**: When a phase crosses 2π, `remainder` abruptly wraps it back to 0. This creates a discontinuous jump that can affect the dynamics, especially when calculating phase differences between oscillators.

2. **Numerical Precision Issues**: The IEEE remainder operation can produce results that are not exactly in [0, 2π) due to floating-point arithmetic, potentially causing small numerical errors to accumulate.

3. **Interaction with Derivatives**: In the Kuramoto model, the interaction term depends on `sin(θⱼ - θᵢ)`. When phases are wrapped discontinuously, the phase differences can suddenly jump by 2π, which shouldn't affect the sine function mathematically but can introduce numerical artifacts in the integration process.

### 3. Surrounding Context

The phase normalization occurs within the RK4 integration step, which is called repeatedly during simulations:

- **Before normalization** (lines 56-69): The RK4 method calculates four derivative estimates (k1, k2, k3, k4) and updates phases
- **After normalization** (line 75): The normalized phases are returned for the next integration step
- **Usage in simulations**: The `run_simulation` function (lines 77-106) calls `rk4_step` multiple times during both transient and measurement phases

### 4. How Phases Are Used Throughout the Codebase

1. **Kuramoto ODEs** (lines 26-47): Phases are used to calculate sine and cosine values for the coupling interactions:
   ```python
   sin_thetas = cp.sin(thetas)
   cos_thetas = cp.cos(thetas)
   ```

2. **Order Parameter Calculation** (lines 99-100): Phases are used to compute the synchronization measure:
   ```python
   exp_thetas = cp.exp(1j * thetas_current)
   r = cp.abs(cp.mean(exp_thetas))
   ```

3. **Visualization** (in `2.py`): Phases are used for coloring nodes in network visualizations, with various transformations to map phase values to colors.

### Potential Issues with Current Implementation

1. **Discontinuous Dynamics**: The abrupt wrapping can cause artificial perturbations in the system dynamics, especially near the synchronization transition.

2. **Numerical Instability**: Near the wrapping boundary, small numerical errors can cause oscillators to appear on opposite sides of the phase circle when they should be close together.

3. **Integration Accuracy**: The discontinuity introduced by `remainder` can reduce the accuracy of the RK4 integration scheme, which assumes smooth dynamics.

### Alternative Approaches

Instead of using `cp.remainder`, consider:

1. **Modulo Operation**: Using `thetas = thetas % (2 * np.pi)` which provides more predictable wrapping behavior
2. **No Normalization**: Since trigonometric functions are periodic, normalization might not be necessary at every step
3. **Conditional Normalization**: Only normalize when phases exceed a certain threshold (e.g., |θ| > 10π)
4. **Smooth Wrapping**: Implement a smooth transition function near the boundaries

The comment on line 71 indicates the normalization is meant to "avoid numerical problems," but the current implementation might actually introduce more issues than it solves in the context of the Kuramoto model dynamics.

## Summary of Phase Normalization in the Codebase

Based on my search through the codebase, I found the following information about phase normalization and handling:

### 1. **Current Phase Normalization Approaches**

- **In `utils.py` line 73**: Uses `cp.remainder(thetas, two_pi, out=thetas)` for phase normalization after each RK4 integration step
- **In `2.py` line 331**: Uses modulo operator `% (2 * np.pi)` for calculating relative phases in the color mapping function
- **In `2.py` line 347**: Uses modulo operator `% (2 * np.pi)` for mean color calculation

### 2. **Phase Difference Calculations**

- **In `utils.py` lines 39-41**: Phase differences are calculated using the trigonometric identity:
  ```python
  # sum(sin(theta_j - theta_i)) = cos(theta_i)*sum(sin(theta_j)) - sin(theta_i)*sum(cos(theta_j))
  interactions = cp.cos(thetas) * sum_sin - cp.sin(thetas) * sum_cos
  ```
  This avoids explicitly computing phase differences and creating an NxN matrix.

- **In `3.py` line 61**: Effective frequencies are calculated as:
  ```python
  effective_freqs = (thetas_final - thetas_start) / T_MEASURE
  ```
  This is a direct phase difference without normalization.

### 3. **Comments About Phase Wrapping Issues**

- **In `utils.py` line 71**: Comment states "Normalizar fases al rango [0, 2π] para evitar problemas numéricos"
- The task description mentions that `cp.remainder` can cause discontinuities and suggests using `thetas % (2*np.pi)` instead

### 4. **No Tests Found**

There are no explicit tests for phase behavior or normalization in the codebase.

### 5. **Key Observations**

1. **Inconsistent normalization methods**: The code uses `cp.remainder()` in the GPU simulation code but `%` operator in the visualization/analysis code.

2. **Phase difference handling**: The main simulation cleverly avoids computing explicit phase differences by using trigonometric identities, which naturally handles phase wrapping.

3. **Potential issue in effective frequency calculation**: In `3.py`, the effective frequency calculation `(thetas_final - thetas_start) / T_MEASURE` doesn't account for phase wrapping, which could lead to incorrect results if phases wrap around during the measurement period.

4. **The main concern**: The use of `cp.remainder()` vs `%` operator. While both should theoretically give the same result for positive angles, `cp.remainder()` follows the IEEE remainder specification which can behave differently near boundaries and with negative values.

The task correctly identifies a potential issue with using `cp.remainder()` for phase normalization, as it could introduce discontinuities or unexpected behavior compared to the standard modulo operation used elsewhere in the codebase.