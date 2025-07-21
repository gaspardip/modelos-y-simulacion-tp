# Project: Kuramoto Model Synchronization Analysis

Computational physics research project analyzing synchronization phenomena in complex networks using the Kuramoto model with GPU acceleration.

## Features
- GPU-accelerated simulations of coupled oscillators (N=10,000 nodes)
- Compares synchronization in complete graphs vs scale-free networks 
- Analyzes phase transitions from disorder to synchronization
- Statistical analysis of critical coupling thresholds
- Automated pipeline for comprehensive network analysis

## Tech Stack
- Python 3.12 with CuPy for GPU computing
- NetworkX for graph operations
- NumPy, Matplotlib, scikit-learn
- LaTeX for academic documentation

## Structure
- src/: Main analysis scripts (1.py-4.py) and utils.py
- img/: Generated plots and visualizations
- todos/: Task management
- tp.tex: LaTeX paper documenting results

## Architecture
- Core simulation in utils.py using Runge-Kutta integration
- GPU-optimized sparse matrix operations
- Separate scripts for different analysis aspects
- Shared constants and utilities

## Commands
- Build: pip install -r requirements.txt
- Test: No formal test suite configured
- Lint: pylint src/*.py or flake8 src/*.py
- Dev/Run: python src/[1-4].py (requires CUDA GPU)
- LaTeX: pdflatex tp.tex

## Testing
No testing framework currently set up. Recommend adding pytest for unit and integration tests of numerical functions.