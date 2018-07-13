from dataclasses import dataclass, field
import numpy as np
from sklearn.mixture.gaussian_mixture import _compute_precision_cholesky
from sklearn.mixture.gaussian_mixture import _compute_log_det_cholesky


@dataclass
class Gaussian:
    mean: np.array  # (..., D)
    covariance: np.array  # (..., D, D)
    precision_cholesky: np.array = field(init=False)  # (..., D, D)
    log_det_precision: np.array = field(init=False)  # (...,)

    def __post_init__(self):
        D = self.mean.shape[-1]
        c = np.reshape(self.covariance, (-1, D, D))
        pc = _compute_precision_cholesky(c, 'full')
        self.precision_cholesky = np.reshape(pc, self.covariance.shape)
        self.log_det_precision = _compute_log_det_cholesky(pc, 'full', D)

    def log_pdf(self, x):
        """Gets used by e.g. the GMM.

        Args:
            x: Shape (..., N, D)

        Returns:

        """
        difference = x - self.mean[..., None, :]
        white_x = np.einsum(
            '...dD,...nD->...nd',
            self.precision_cholesky,
            difference
        )
        return (
                - 1 / 2 * np.log(2 * np.pi) + self.log_det_precision
                - 1 / 2 * np.einsum('...nd,...nd->...n', white_x, white_x)
        )


@dataclass
class DiagonalGaussian:
    mean: np.array  # (..., D)
    covariance: np.array  # (..., D)

    def log_pdf(self, x):
        raise NotImplementedError


@dataclass
class SphericalGaussian:
    mean: np.array  # (..., D)
    covariance: np.array  # (...,)

    def log_pdf(self, x):
        raise NotImplementedError


class GaussianTrainer:
    def fit(self, x, saliency=None, covariance_type="full"):
        """

        Args:
            x: Shape (..., N, D)
            saliency: Importance weighting for each observation, shape (..., N)
            covariance_type: Either 'full', 'diagonal', or 'spherical'

        Returns:

        """
        assert np.isrealobj(x), x.dtype
        return self._fit(x, saliency=saliency, covariance_type=covariance_type)

    def _fit(self, x, saliency, covariance_type):
        dimension = x.shape[-1]

        if saliency is None:
            denominator = np.array(x.shape[-2])
        else:
            denominator = np.sum("...n->...", saliency)

        mean = np.einsum("...nd->...d", x) / denominator[..., None]
        difference = x - mean[..., None, :]

        if covariance_type == "full":
            operation = "...nd,...nD->...dD"
            denominator = denominator[..., None, None]
            model_cls = Gaussian
        elif covariance_type == "diagonal":
            operation = "...nd,...nd->...d"
            denominator = denominator[..., None]
            model_cls = DiagonalGaussian
        elif covariance_type == "spherical":
            operation = "...nd,...nd->..."
            model_cls = DiagonalGaussian
            denominator = denominator * dimension
        else:
            raise ValueError(f"Unknown covariance type '{covariance_type}'.")

        if saliency is None:
            covariance = np.einsum(operation, difference, difference)
        else:
            operation = "...n," + operation
            covariance = np.einsum(operation, saliency, difference, difference)

        covariance /= denominator
        return model_cls(mean=mean, covariance=covariance)
