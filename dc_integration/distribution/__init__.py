"""Rationale:

Code for each kind of distribution lives in its own file, e.g. `gmm.py`.

Each file contains a trainer with a fit function. It may keep shared state
among each training, i.e. lookup tables.

Each file contains at least one dataclass, e.g. `Gaussian`. It stores the
parameters and provides at least a predict function.


...
"""
from .gaussian import (
    Gaussian,
    DiagonalGaussian,
    SphericalGaussian,
    GaussianTrainer,
)
from .gmm import GMM, GMMTrainer
from .circular_symmetric_gaussian import CircularSymmetricGaussian
from .complex_angular_central_gaussian import (
    ComplexAngularCentralGaussian,
    ComplexAngularCentralGaussianTrainer,
)
from .cacgmm import CACGMM, CACGMMTrainer
