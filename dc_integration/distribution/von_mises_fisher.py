from dataclasses import dataclass
from scipy.special import ive
import numpy as np


@dataclass
class VonMisesFisherParameters:
    mu: np.array = None
    kappa: np.array = None


class VonMisesFisher:
    """
    All inputs should be of type ndarray.

    Expected shapes:
    x: (D, N) SOLL (..., D)
    mu: (K, D) SOLL (..., D)
    kappa: (K,) SOLL (...)
    """

    @classmethod
    def norm(cls, kappa, D):
        return np.exp(cls.log_norm(kappa, D))

    @classmethod
    def pdf(cls, x, mu, kappa):
        return np.exp(cls.log_pdf(x, mu, kappa))

    @classmethod
    def log_norm(cls, kappa, D):
        kappa = np.clip(kappa, 1e-10, np.inf)
        return (
            (D / 2) * np.log(2 * np.pi) + np.log(ive(D / 2 - 1, kappa))
            + np.abs(kappa) - (D / 2 - 1) * np.log(kappa)
        )

    @classmethod
    def log_pdf(cls, x, mu, kappa):
        """ Logarithm of probability density function.

        Args:
            x: Observations with shape (..., D), i.e. (1, N, D).
            mu: Mean direction with shape (..., D), i.e. (K, 1, D).
            kappa: Concentration parameter with shape (...), i.e. (K, 1).

        Returns: Probability density with shape (...), i.e. (K, N).
        """
        # For now, we assume that the caller does proper expansion
        D = x.shape[-1]
        assert x.ndim == mu.ndim
        assert x.ndim - 1 == kappa.ndim
        result = np.einsum('...d,...d', x, mu)
        result *= kappa
        result -= cls.log_norm(kappa, D)
        return result

    @classmethod
    def fit(
            cls, x, weights=None,
            min_concentration=0, max_concentration=500
    ) -> VonMisesFisherParameters:
        """ Fits a von Mises Fisher distribution.

        Broadcasting (for sources) has to be done outside this function.

        Args:
            x: Observations with shape (..., N, D)
            weights: Either None or weights with shape (..., N)
        """
        D = x.shape[-1]
        weights = np.ones(x.shape[:-1]) if weights is None else weights

        r = np.sum(x * weights[..., None], axis=-2)
        norm = np.linalg.norm(r, axis=-1)
        mu = r / norm[..., None]
        r_bar = norm / np.sum(weights, axis=-1)

        # From [Banerjee2005vMF]
        kappa = (r_bar * D - r_bar ** 3) / (1 - r_bar ** 2)
        kappa = np.clip(kappa, min_concentration, max_concentration)

        return mu, kappa
