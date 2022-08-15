# DiffDRR

DiffDRR is a PyTorch-based DRR generator that provides

1. Auto-differentiable DRR syntheisis
2. GPU-accelerated rendering
3. A pure Python implementation.

## Installation

```
git clone https://github.com/v715/DiffDRR
conda env create -f environment.yaml
conda activate DiffDRR
```

## Usage

```Python
import matplotlib.pyplot as plt
import numpy as np
import torch

from src import read_dicom, DRR
from src.visualization import plot_drr
```


## Mechanics

DiffDRR reformulates Siddon's method [[1]](https://aapm.onlinelibrary.wiley.com/doi/abs/10.1118/1.595715), the canonical algorithm for calculating the radiologic path of an X-ray through a volume, as a series of vectorized tensor operations.
This version of the algorithm is easily implemented in tensor algebra libraries like PyTorch to achieve a fast auto-differentiable DRR generator.
