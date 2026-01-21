# SparseRecovery

Implementation of algorithms for high-dimensional sparse recovery from function samples, including decoders, guarantees, and instance optimality analysis.

## Overview

This repository contains the code accompanying the paper:

**"High-dimensional sparse recovery from function samples: Decoders, guarantees and instance optimality"**  
📄 [arXiv:2503.16209](https://arxiv.org/abs/2503.16209)

The code implements several sparse recovery algorithms (OMP, LASSO, CoSaMP) applied to various test functions, with analysis of their recovery performance and optimality properties.

## Repository Structure

```
├── algorithms/          # Core recovery algorithm implementations
├── best_m-term/         # Notebooks for best m-term approximation 
├── results/             # Txt files with output values
├── test_notebooks/      # Jupyter notebooks for experiments
├── OMP_Fun*.py          # Orthogonal Matching Pursuit experiments
├── LASSO_Fun*.py        # LASSO experiments
└── COSaMP_Fun*.py       # Compressive Sampling Matching Pursuit
```

## Algorithms Implemented

- **Orthogonal Matching Pursuit (OMP)**: Greedy sparse approximation
- **LASSO**: L1-regularized L2-norm minimization
- **CoSaMP**: Compressive Sampling Matching Pursuit

Multiple test functions (Fun1, Fun2, Fun3, Fun4) are used to evaluate algorithm performance across different problem structures.

## Usage

Each Python script runs experiments for specific algorithm-function combinations:

```bash
python OMP_Fun1.py    # Run OMP on Function 1
python LASSO_Fun2.py  # Run LASSO on Function 2
```

Jupyter notebooks in `test_notebooks/` provide interactive exploration and visualization of results.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{sparserecovery2025,
  title={High-dimensional sparse recovery from function samples: Decoders, guarantees and instance optimality},
  author={[Authors]},
  year={2025},
  eprint={2503.16209},
  archivePrefix={arXiv}
}
```

## Requirements

- Python 3.x
- NumPy
- SciPy
- Jupyter (for notebooks)
- Pykeops
- Deepinv
- Tqdm

## License

Please refer to the LICENSE file for terms of use.
