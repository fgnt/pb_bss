"""
References:
    Robert G Gallagar
    Circularly-Symmetric Gaussian random vectors
    http://www.rle.mit.edu/rgallager/documents/CircSymGauss.pdf

    Wikipedia
    https://en.wikipedia.org/wiki/Complex_normal_distribution#Circularly-symmetric_normal_distribution

    Andersen et al., Lecture Notes in Statistics, Linear and Graphical Models
    for the Multivariate Complex Normal Distribution
    Theoreme 2.10
    https://link.springer.com/content/pdf/10.1007%2F978-1-4612-4240-6.pdf
"""
from dataclasses import dataclass

import numpy as np
from pb_bss.distribution.utils import _ProbabilisticModel
from pb_bss.utils import is_broadcast_compatible


@dataclass
class ComplexCircularSymmetricGaussian(_ProbabilisticModel):
    covariance: np.array  # (..., D, D)

    def log_pdf(self, y):
        """Gets used by e.g. the GMM.

        Args:
            y: Shape (..., N, D)

        Returns:

        """
        D = self.covariance.shape[-1]
        return (
            - D * np.log(np.pi)
            - np.linalg.slogdet(self.covariance)[-1][..., None]
            - np.einsum(
                '...nd,...nd->...n',
                y.conj(),
                np.squeeze(np.linalg.solve(
                    self.covariance[..., None, :, :],
                    y[..., :, None]),
                    axis=-1
                )
            ).real
        )

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


class ComplexCircularSymmetricGaussianTrainer:
    def fit(self, y, saliency=None, covariance_type="full"):
        """

        Args:
            y: Shape (..., N, D)
            saliency: Importance weighting for each observation, shape (..., N)
            covariance_type: Either 'full', 'diagonal', or 'spherical'

        Returns:

        """
        assert np.iscomplexobj(y), y.dtype
        if saliency is not None:
            assert is_broadcast_compatible(y.shape[:-1], saliency.shape), (
                y.shape, saliency.shape
            )
        return self._fit(y, saliency=saliency, covariance_type=covariance_type)

    def _fit(self, y, saliency, covariance_type):
        if saliency is None:
            denominator = np.array(y.shape[-2])
        else:
            denominator = np.maximum(
                np.einsum("...n->...", saliency),
                np.finfo(y.dtype).tiny
            )

        if covariance_type == "full":
            operation = "...nd,...nD->...dD"
            denominator = denominator[..., None, None]
            model_cls = ComplexCircularSymmetricGaussian
        else:
            raise ValueError(f"Unknown covariance type '{covariance_type}'.")

        if saliency is None:
            covariance = np.einsum(operation, y, y.conj())
        else:
            operation = "...n," + operation
            covariance = np.einsum(operation, saliency, y, y.conj())
        covariance /= denominator
        return model_cls(covariance=covariance)
