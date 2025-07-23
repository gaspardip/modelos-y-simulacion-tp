# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a computational physics research project analyzing synchronization phenomena in complex networks using the Kuramoto model. The codebase implements GPU-accelerated simulations to study phase transitions from disorder to synchronization in large networks (N>=10,000 nodes), comparing complete graphs versus scale-free networks.

## Core Architecture

### Two-Layer GPU Architecture
- **`src/kernel.py`**: Low-level CUDA kernels with architecture-adaptive optimization
  - `warp_optimized_kernel`: Uses warp shuffle reductions (best for Ampere/Hopper GPUs)
  - `memory_hierarchy_kernel`: Uses shared memory optimization (modern GPUs)
  - `baseline_batch_kernel`: Standard implementation (legacy compatibility)
  - Automatic GPU architecture detection and kernel selection
  - All kernels implement **corrected RK4 integration** with proper neighbor recalculation at each stage

- **`src/utils.py`**: High-level simulation orchestration
  - Network generation (scale-free, complete graphs)
  - Batch simulation management with time-weighted averaging
  - CSR sparse matrix optimizations for memory efficiency
  - Integration with kernel.py for GPU acceleration

### Critical RK4 Implementation
All GPU kernels were recently fixed to implement **correct Runge-Kutta 4th order integration**. The key insight is that neighbor influence terms must be recalculated at each of the 4 RK4 stages using updated phase values, not reused. This fixes a fundamental physics bug that was causing incorrect critical coupling (Kc) values.

### Analysis Pipeline (Four Scripts)
- **`1.py`**: Quantitative analysis - generates r vs K plots comparing network topologies
- **`2.py`**: Visual analysis - creates network visualizations in different synchronization regimes
- **`3.py`**: Deep analysis - studies the relationship between structural hubs and dynamic cores
- **`4.py`**: Statistical validation - runs Monte Carlo analysis of critical thresholds

## Development Commands

### Environment Setup
**CRITICAL**: You must activate the Python virtual environment before running any code:
```bash
# Activate the virtual environment first
source venv/bin/activate
```

### Running Simulations
```bash
# Execute analysis pipeline in order
cd src/
python 1.py  # Quantitative r vs K analysis
python 2.py  # Network visualization analysis
python 3.py  # Hub vs core comparison
python 4.py  # Statistical validation (long-running)
```

### Testing GPU Kernels
```bash
# Test corrected kernels (requires CuPy)
source venv/bin/activate
cd src/
python -c "
from kernel import batch_kuramoto_simulation, detect_gpu_architecture
print('GPU tier:', detect_gpu_architecture())
# Run small test simulation
"
```

## Key Configuration Parameters

Located in `src/utils.py`:
- `N = 10000`: Network size (large-scale simulations)
- `M_SCALE_FREE = 5`: Edges per node in Barab�si-Albert networks
- `DT = 0.01`: RK4 integration timestep
- `T_TRANSIENT = 5`: Transient time to discard
- `T_MEASURE = 10`: Measurement period for order parameter
- `K_VALUES_SWEEP`: Coupling strength range [0, 5] with 50 points
- `R_THRESHOLD = 0.5`: Synchronization threshold

## Memory Optimization Strategy

The codebase uses sparse matrix operations to handle large networks efficiently:
- Dense adjacency matrices: O(N^2) memory infeasible for N=10,000
- CSR sparse format: O(E) memory where E is number of edges
- Trigonometric identity optimization: sin(|i - |j) = cos(|j)sin(|i) - sin(|j)cos(|i)
- Batch processing of multiple K values simultaneously for GPU efficiency

## GPU Performance Notes

- **Verified corrected physics**: All kernels now implement proper RK4 with neighbor recalculation
- **Architecture adaptive**: Automatically selects optimal kernel based on GPU capability
- **Expected performance**: N=10,000 networks should run in minutes on modern GPUs (RTX 4090)
- **Fallback mechanism**: Graceful degradation from warp-optimized � memory-optimized � baseline kernels

## Common Issues

### CuPy Installation
Project requires GPU environment with CuPy. If running locally, ensure CUDA toolkit matches CuPy version.

### Memory Errors
Large networks may exceed GPU memory. Reduce `N` or implement batch processing if needed.

### Incorrect Kc Values
If getting uniform Kc H 0.102 instead of expected range 1-3 for scale-free networks, ensure using the corrected kernels in `kernel.py` (not the old implementation in `utils.py`).

## Research Context

This project documents the evolution from initial hypothesis (hubs facilitate synchronization) to refined understanding (hubs act as catalysts but aren't the most synchronized nodes). The technical evolution includes migration from CPU scipy to GPU CuPy, from Euler to RK4 integration, and from dense to sparse matrix operations. The corrected RK4 implementation was the final critical fix ensuring physical accuracy.