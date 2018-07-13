"""Rationale:

Code for each kind of distribution lives in its own file, e.g. `gmm.py`.

Each file contains a trainer with a fit function. It may keep shared state
among each training, i.e. lookup tables.

Each file contains at least one dataclass, e.g. `Gaussian`. It stores the
parameters and provides at least a predict function.


...
"""
from .complex_angular_central_gaussian import (
    ComplexAngularCentralGaussian,
    ComplexAngularCentralGaussianParameters,
)
from .cacgmm import (
    ComplexAngularCentralGaussianMixtureModel,
    ComplexAngularCentralGaussianMixtureModelParameters,
)
from .complex_watson import (
    ComplexWatson,
    ComplexWatsonParameters,
)
from .cwmm import (
    ComplexWatsonMixtureModel,
    ComplexWatsonMixtureModelParameters,
)
