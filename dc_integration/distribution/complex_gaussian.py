"""
Proper complex Gaussian distribution
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class ComplexGaussianParameters:
    covariance: np.array = None
    precision: np.array = None
    determinant: np.array = None
