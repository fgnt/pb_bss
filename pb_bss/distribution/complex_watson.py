import math
from dataclasses import dataclass

from scipy.interpolate import interp1d
from scipy.special import hyp1f1
from cached_property import cached_property

import numpy as np

from pb_bss.distribution.utils import _ProbabilisticModel
from pb_bss.utils import is_broadcast_compatible

from pb_bss.utils import get_power_spectral_density_matrix, get_pca


def normalize_observation(observation):
    """

    Args:
        observation: (..., N, D)

    Returns:
        normalized observation (..., N, D)
    """
    # ToDo: Should the dimensions be swapped like in cacg for speed?
    return observation / np.maximum(
        np.linalg.norm(observation, axis=-1, keepdims=True),
        np.finfo(observation.dtype).tiny,
    )


@dataclass
class ComplexWatson(_ProbabilisticModel):
    """
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> scales = [
    ...     np.arange(0, 0.01, 0.001),
    ...     np.arange(0, 20, 0.01),
    ...     np.arange(0, 100, 1)
    ... ]
    >>> functions = [
    ...     ComplexWatson.log_norm_low_concentration,
    ...     ComplexWatson.log_norm_medium_concentration,
    ...     ComplexWatson.log_norm_high_concentration
    ... ]
    >>>
    >>> f, axis = plt.subplots(1, 3)
    >>> for ax, scale in zip(axis, scales):
    ...     result = [fn(scale, 6) for fn in functions]
    ...     _ = [ax.plot(scale, np.log(r), '--') for r in result]
    ...     _ = ax.legend(['low', 'middle', 'high'])
    >>> _ = plt.show()
    """

    mode: np.array = None  # Shape (..., D)
    concentration: np.array = None  # Shape (...)

    def pdf(self, y):
        """ Calculates pdf function.

        Args:
            y: Assumes shape (..., D).
            loc: Mode vector. Assumes corresponding shape (..., D).
            scale: Concentration parameter with shape (...).

        Returns:
        """
        return np.exp(self.log_pdf(y))

    def log_pdf(self, y):
        """ Calculates logarithm of pdf function.

        Args:
            y: Assumes shape (..., D).
            loc: Mode vector. Assumes corresponding shape (..., D).
            scale: Concentration parameter with shape (...).

        Returns:
        """
        result = np.einsum("...d,...d", y, self.mode[..., None, :].conj())
        result = result.real ** 2 + result.imag ** 2
        result *= self.concentration[..., None]
        result -= self.log_norm()[..., None]
        return result

    @staticmethod
    def log_norm_low_concentration(scale, dimension):
        """ Calculates logarithm of pdf function.
        Good at very low concentrations but starts to drop of at 20.
        """
        scale = np.asfarray(scale)
        shape = scale.shape
        scale = scale.ravel()

        # Mardia1999Watson Equation 4, Taylor series
        b_range = range(dimension, dimension + 20 - 1 + 1)
        b_range = np.asarray(b_range)[None, :]

        return (
            np.log(2)
            + dimension * np.log(np.pi)
            - np.log(math.factorial(dimension - 1))
            + np.log(1 + np.sum(np.cumprod(scale[:, None] / b_range, -1), -1))
        ).reshape(shape)

    @staticmethod
    def log_norm_medium_concentration(scale, dimension):
        """ Calculates logarithm of pdf function.
        Almost complete range of interest and dimension below 8.
        """
        scale = np.asfarray(scale)
        shape = scale.shape
        scale = scale.flatten()

        # Function is unstable at zero.
        # Scale needs to be float for this to work.
        scale[scale < 1e-2] = 1e-2

        r_range = range(dimension - 2 + 1)
        r = np.asarray(r_range)[None, :]

        # Mardia1999Watson Equation 3
        temp = (
            scale[:, None] ** r
            * np.exp(-scale[:, None])
            / np.asarray([math.factorial(_r) for _r in r_range])
        )

        return (
            np.log(2.)
            + dimension * np.log(np.pi)
            + (1. - dimension) * np.log(scale)
            + scale
            + np.log(1. - np.sum(temp, -1))
        ).reshape(shape)

    @staticmethod
    def log_norm_high_concentration(scale, dimension):
        """ Calculates logarithm of pdf function.
        High concentration above 10 and dimension below 8.
        """
        scale = np.asfarray(scale)
        shape = scale.shape
        scale = scale.ravel()

        return (
            np.log(2.)
            + dimension * np.log(np.pi)
            + (1. - dimension) * np.log(scale)
            + scale
        ).reshape(shape)

    @staticmethod
    def log_norm_1f1(scale, dimension):
        # This is already good.
        # I am unsure if the others are better.
        # In https://github.com/scipy/scipy/issues/2957
        # is denoted that they solved a precision issue

        # With scale > 800 this function makes problems.
        # Normally scale is thresholded by 100
        norm = hyp1f1(1, dimension, scale) * (
            2 * np.pi ** dimension / math.factorial(dimension - 1)
        )
        return np.log(norm)

    @staticmethod
    def log_norm_tran_vu(scale, dimension):
        scale = np.array(scale)
        shape = scale.shape
        scale = scale.ravel()

        log_c_high = (
            np.log(2.)
            + dimension * np.log(np.pi)
            + (1. - dimension) * np.log(scale)
            + scale
        )

        r_range = np.arange(dimension - 2 + 1)
        r = r_range[None, :]
        # Mardia1999Watson Equation 3
        temp = (
            scale[:, None] ** r
            * np.exp(-scale[:, None])
            / np.array([math.factorial(_r) for _r in r_range])
        )

        log_c_medium = log_c_high + (+np.log(1. - np.sum(temp, -1)))

        # Mardia1999Watson Equation 4, Taylor series
        b_range = np.arange(dimension, dimension + 20 - 1 + 1)
        b_range = b_range[None, :]

        log_c_low = (
            np.log(2)
            + dimension * np.log(np.pi)
            - np.log(math.factorial(dimension - 1))
            + np.log(1 + np.sum(np.cumprod(scale[:, None] / b_range, -1), -1))
        )

        # A good boundary between low and medium is kappa = 1/D. This can be
        # motivated by plotting each term for very small values and for
        # values from [0.1, 1].
        # A good boundary between medium and high is when
        # kappa * exp(kappa) < epsilon. Choosing kappa = 100 is sufficient.
        log_c_low[scale >= 1 / dimension] = log_c_medium[
            scale >= 1 / dimension
        ]
        log_c_low[scale >= 100] = log_c_medium[scale >= 100]
        return log_c_low.reshape(shape)

    def log_norm(self):
        return self.log_norm_1f1(self.concentration, self.mode.shape[-1])


class ComplexWatsonTrainer:
    def __init__(
        self, dimension=None, max_concentration=500, spline_markers=1000
    ):
        """

        Args:
            dimension: Feature dimension. If you do not provide this when
                initializing the trainer, it will be inferred when the fit
                function is called.
            max_concentration:
            spline_markers:
        """
        self.dimension = dimension
        self.max_concentration = max_concentration
        self.spline_markers = spline_markers

    @cached_property
    def spline(self):
        """Defines a cubic spline to fit concentration parameter."""
        assert self.dimension is not None, (
            "You need to specify dimension. This can be done at object "
            "instantiation or it can be inferred when using the fit function."
        )
        x = np.logspace(
            -3, np.log10(self.max_concentration), self.spline_markers
        )
        y = self.hypergeometric_ratio(x)

        return interp1d(
            y,
            x,
            kind="quadratic",
            assume_sorted=True,
            bounds_error=False,
            fill_value=(0, self.max_concentration),
        )

    def hypergeometric_ratio(self, concentration):
        eigenvalue = hyp1f1(2, self.dimension + 1, concentration) / (
            self.dimension * hyp1f1(1, self.dimension, concentration)
        )
        return eigenvalue

    def hypergeometric_ratio_inverse(self, eigenvalues):
        """
        This is twice as slow as interpolation with Tran Vu's C-code.

        >>> t = ComplexWatsonTrainer(5)
        >>> t.hypergeometric_ratio_inverse([0, 1/5, 1/5 + 1e-4, 0.9599999, 1])
        """

        return self.spline(eigenvalues)

    def fit(self, y, saliency=None) -> ComplexWatson:
        assert np.iscomplexobj(y), y.dtype
        assert y.shape[-1] > 1
        y = y / np.maximum(
            np.linalg.norm(y, axis=-1, keepdims=True), np.finfo(y.dtype).tiny
        )

        if saliency is not None:
            assert is_broadcast_compatible(y.shape[:-1], saliency.shape), (
                y.shape,
                saliency.shape,
            )

        if self.dimension is None:
            self.dimension = y.shape[-1]
        else:
            assert self.dimension == y.shape[-1], (
                "You initialized the trainer with a different dimension than "
                "you are using to fit a model. Use a new trainer, when you "
                "change the dimension."
            )

        return self._fit(y, saliency=saliency)

    def _fit(self, y, saliency) -> ComplexWatson:
        if saliency is None:
            covariance = np.einsum(
                "...nd,...nD->...dD", y, y.conj()
            )
            denominator = np.array(y.shape[-2])
        else:
            covariance = np.einsum(
                "...n,...nd,...nD->...dD", saliency, y, y.conj()
            )
            denominator = np.einsum("...n->...", saliency)[..., None, None]

        covariance /= denominator
        mode, eigenvalues = get_pca(covariance)
        concentration = self.hypergeometric_ratio_inverse(eigenvalues)
        return ComplexWatson(mode=mode, concentration=concentration)
