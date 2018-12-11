"""
References:
    Robert G Gallagar
    Circularly-Symmetric Gaussian random vectors
    http://www.rle.mit.edu/rgallager/documents/CircSymGauss.pdf

    Wikipedia
    https://en.wikipedia.org/wiki/Complex_normal_distribution#Circularly-symmetric_normal_distribution
"""
from dataclasses import dataclass

import numpy as np
from pb_bss.distribution.utils import _ProbabilisticModel


@dataclass
class ComplexCircularSymmetricGaussian(_ProbabilisticModel):
    covariance: np.array  # (..., D, D)

    def sample(self, size):
        """

        Args:
            size: Using size (...,) will yield shape (..., D). This is analogue
                to `np.random.multivariate_normal`.

        Returns:

        """
        if self.covariance.ndim > 2:
            # TODO: What is the correct generalization?
            raise NotImplementedError(
                "Not quite clear how the correct broadcasting would look like."
            )
        D = self.covariance.shape[-1]
        real = np.random.normal(size=(*size, D))
        imag = np.random.normal(size=(*size, D))
        x = real + 1j * imag
        x /= np.sqrt(2)
        cholesky = np.linalg.cholesky(self.covariance)
        x = (cholesky @ x.T).T
        return x
