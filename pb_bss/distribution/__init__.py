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
from .gmm import GMM, GMMTrainer, BinaryGMM, BinaryGMMTrainer
from .von_mises_fisher import VonMisesFisher, VonMisesFisherTrainer
from .complex_circular_symmetric_gaussian import (
    ComplexCircularSymmetricGaussian,
    ComplexCircularSymmetricGaussianTrainer
)
from .complex_angular_central_gaussian import (
    ComplexAngularCentralGaussian,
    ComplexAngularCentralGaussianTrainer,
)
from .complex_watson import ComplexWatson, ComplexWatsonTrainer
from .vmfmm import VMFMM, VMFMMTrainer
from .vmfcacgmm import VMFCACGMM, VMFCACGMMTrainer
from .gcacgmm import GCACGMM, GCACGMMTrainer
from .cacgmm import CACGMM, CACGMMTrainer, sample_cacgmm, normalize_observation
from .cwmm import CWMM, CWMMTrainer
from .cbmm import CBMM, CBMMTrainer

from . import utils
from . import mixture_model_utils
