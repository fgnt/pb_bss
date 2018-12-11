"""
@Article{Banerjee2005vMF,
  author  = {Banerjee, Arindam and Dhillon, Inderjit S and Ghosh, Joydeep and Sra, Suvrit},
  title   = {Clustering on the unit hypersphere using von {M}ises-{F}isher distributions},
  journal = {Journal of Machine Learning Research},
  year    = {2005},
  volume  = {6},
  number  = {Sep},
  pages   = {1345--1382},
}

@article{Wood1994Simulation,
  title={Simulation of the von Mises Fisher distribution},
  author={Wood, Andrew TA},
  journal={Communications in statistics-simulation and computation},
  volume={23},
  number={1},
  pages={157--164},
  year={1994},
  publisher={Taylor \& Francis}
}
"""
from dataclasses import dataclass
from scipy.special import ive
import numpy as np
from pb_bss.distribution.utils import _ProbabilisticModel
from pb_bss.utils import is_broadcast_compatible


@dataclass
class VonMisesFisher(_ProbabilisticModel):
    mean: np.array  # (..., D)
    concentration: np.array  # (...,)

    def log_norm(self):
        """Is fairly stable, when concentration > 1e-10."""
        D = self.mean.shape[-1]
        return (
            (D / 2) * np.log(2 * np.pi)
            + np.log(ive(D / 2 - 1, self.concentration))
            + (
                np.abs(self.concentration)
                - (D / 2 - 1) * np.log(self.concentration)
            )
        )

    def sample(self, size):
        """
        Sampling according to [Wood1994Simulation].

        Args:
            size:

        Returns:

        """
        raise NotImplementedError(
            'A good implementation can be found in libdirectional: '
            'https://github.com/libDirectional/libDirectional/blob/master/lib/distributions/Hypersphere/VMFDistribution.m#L239'
        )

    def norm(self):
        return np.exp(self.log_norm)

    def log_pdf(self, y):
        """ Logarithm of probability density function.

        Args:
            y: Observations with shape (..., D), i.e. (1, N, D).

        Returns: Log-probability density with properly broadcasted shape.
        """
        y = y / np.maximum(
            np.linalg.norm(y, axis=-1, keepdims=True), np.finfo(y.dtype).tiny
        )
        result = np.einsum("...d,...d", y, self.mean[..., None, :])
        result *= self.concentration[..., None]
        result -= self.log_norm()[..., None]
        return result

    def pdf(self, y):
        """ Probability density function.

        Args:
            y: Observations with shape (..., D), i.e. (1, N, D).

        Returns: Probability density with properly broadcasted shape.
        """
        return np.exp(self.log_pdf(y))


class VonMisesFisherTrainer:
    def fit(
        self, y, saliency=None, min_concentration=1e-10, max_concentration=500
    ) -> VonMisesFisher:
        """ Fits a von Mises Fisher distribution.

        Broadcasting (for sources) has to be done outside this function.

        Args:
            y: Observations with shape (..., N, D)
            saliency: Either None or weights with shape (..., N)
            min_concentration:
            max_concentration:
        """
        assert np.isrealobj(y), y.dtype
        y = y / np.maximum(
            np.linalg.norm(y, axis=-1, keepdims=True), np.finfo(y.dtype).tiny
        )
        if saliency is not None:
            assert is_broadcast_compatible(y.shape[:-1], saliency.shape), (
                y.shape,
                saliency.shape,
            )
        return self._fit(
            y,
            saliency=saliency,
            min_concentration=min_concentration,
            max_concentration=max_concentration,
        )

    def _fit(
        self, y, saliency, min_concentration, max_concentration
    ) -> VonMisesFisher:

        D = y.shape[-1]

        if saliency is None:
            saliency = np.ones(y.shape[:-1])

        # [Banerjee2005vMF] Equation 2.4
        r = np.einsum("...n,...nd->...d", saliency, y)
        norm = np.linalg.norm(r, axis=-1)
        mean = r / np.maximum(norm, np.finfo(y.dtype).tiny)[..., None]

        # [Banerjee2005vMF] Equation 2.5
        r_bar = norm / np.sum(saliency, axis=-1)

        # [Banerjee2005vMF] Equation 4.4
        concentration = (r_bar * D - r_bar ** 3) / (1 - r_bar ** 2)
        concentration = np.clip(
            concentration, min_concentration, max_concentration
        )
        return VonMisesFisher(mean=mean, concentration=concentration)
