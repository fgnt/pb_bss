from dataclasses import dataclass

import numpy as np


@dataclass
class Gaussian:
    mean: np.array = None  # (..., D)
    covariance: np.array = None  # (..., D, D)

    def predict(self):
        pass


@dataclass
class DiagonalGaussian:
    mean: np.array = None  # (..., D)
    covariance: np.array = None  # (..., D)

    def predict(self):
        pass


@dataclass
class ScaledIdentityGaussian:
    mean: np.array = None  # (..., D)
    covariance: np.array = None  # (...,)

    def predict(self):
        pass


class Trainer:
    def fit(self, covariance_type='full'):
        pass
