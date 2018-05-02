from dataclasses import dataclass

import numpy as np


@dataclass
class ComplexAngularCentralGaussianParameters:
    covariance: np.array = None
    precision: np.array = None
    determinant: np.array = None

    def predict(self):
        pass


class ComplexAngularCentralGaussian:
    def fit(
            self, Y, saliency=None
    ) -> ComplexAngularCentralGaussianParameters:
        """

        Args:
            Y: Complex observation with shape (..., T, D)
            saliency: Weight for each observation (..., T)

        Returns:

        """
        pass
