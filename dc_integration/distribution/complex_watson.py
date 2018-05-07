import math
from dataclasses import dataclass

from scipy.interpolate import interp1d
from scipy.special import hyp1f1

import numpy as np

from dc_integration.distribution.util import (
    _Parameter,
)


@dataclass
class ComplexWatsonParameters(_Parameter):
    mode: np.array = None
    concentration: np.array = None


class ComplexWatson:
    """
    >>> # import dc_integration.distributions.complex_watson as cw
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

    @staticmethod
    def pdf(x, loc, scale):
        """ Calculates pdf function.

        Args:
            x: Assumes shape (..., D).
            loc: Mode vector. Assumes corresponding shape (..., D).
            scale: Concentration parameter with shape (...).

        Returns:
        """
        return np.exp(ComplexWatson.log_pdf(x, loc, scale))

    @staticmethod
    def log_pdf(x, loc, scale):
        """ Calculates logarithm of pdf function.

        Args:
            x: Assumes shape (..., D).
            loc: Mode vector. Assumes corresponding shape (..., D).
            scale: Concentration parameter with shape (...).

        Returns:
        """
        # For now, we assume that the caller does proper expansion
        assert x.ndim == loc.ndim
        assert x.ndim - 1 == scale.ndim

        result = np.einsum('...d,...d', x, loc.conj())
        result = result.real ** 2 + result.imag ** 2
        result *= scale
        result -= ComplexWatson.log_norm(scale, x.shape[-1])
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
            np.log(2) + dimension * np.log(np.pi) -
            np.log(math.factorial(dimension - 1)) +
            np.log(1 + np.sum(np.cumprod(scale[:, None] / b_range, -1), -1))
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
        temp = scale[:, None] ** r * np.exp(-scale[:, None]) / \
               np.asarray([math.factorial(_r) for _r in r_range])

        return (
            np.log(2.) + dimension * np.log(np.pi) +
            (1. - dimension) * np.log(scale) + scale +
            np.log(1. - np.sum(temp, -1))
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
            np.log(2.) + dimension * np.log(np.pi) +
            (1. - dimension) * np.log(scale) + scale
        ).reshape(shape)

    @staticmethod
    def log_norm_1F1(scale, dimension):
        # This is already good.
        # I am unsure if the others are better.
        # In https://github.com/scipy/scipy/issues/2957
        # is denoted that they solved a precision issue

        # With scale > 800 this function makes problems.
        # Normally scale is thresholded by 100
        norm = hyp1f1(1, dimension, scale) * (
                2*np.pi**dimension / math.factorial(dimension-1)
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
        temp = scale[:, None] ** r * np.exp(-scale[:, None]) / \
               np.array([math.factorial(_r) for _r in r_range])

        log_c_medium = log_c_high + (
            + np.log(1. - np.sum(temp, -1))
        )

        # Mardia1999Watson Equation 4, Taylor series
        b_range = np.arange(dimension, dimension + 20 - 1 + 1)
        b_range = b_range[None, :]

        log_c_low = (
                np.log(2)
                + dimension * np.log(np.pi)
                - np.log(math.factorial(dimension - 1))
                + np.log(
                    1
                    + np.sum(np.cumprod(scale[:, None] / b_range, -1), -1)
                )
        )

        # A good boundary between low and medium is kappa = 1/D. This can be
        # motivated by plotting each term for very small values and for
        # values from [0.1, 1].
        # A good boundary between medium and high is when
        # kappa * exp(kappa) < epsilon. Choosing kappa = 100 is sufficient.
        log_c_low[scale >= 1/dimension] = log_c_medium[scale >= 1/dimension]
        log_c_low[scale >= 100] = log_c_medium[scale >= 100]
        return log_c_low.reshape(shape)

    log_norm = log_norm_1F1

    def __init__(self, dimension, max_concentration=100, spline_markers=100):
        self.dimension = dimension
        self.max_concentration = max_concentration
        self.spline_markers = spline_markers
        self.spline = self._get_spline()

    def _get_spline(self):
        """Defines a qubic spline to fit concentration parameter."""
        x = np.logspace(
            -3, np.log10(self.max_concentration), self.spline_markers
        )
        y = (
            hyp1f1(2, self.dimension + 1, x)
            / (self.dimension * hyp1f1(1, self.dimension, x))
        )
        return interp1d(
            y, x,
            kind='quadratic', assume_sorted=True, bounds_error=False,
            fill_value=(0, self.max_concentration)
        )

    def hypergeometric_ratio_inverse(self, eigenvalues):
        """
        This is twice as slow as interpolation with Tran Vu's C-code.

        >>> cw = ComplexWatson(5)
        >>> cw.hypergeometric_ratio_inverse([0, 1/5, 1/5 + 1e-4, 0.9599999, 1])
        """

        return self.spline(eigenvalues)

    def fit(self) -> ComplexWatsonParameters:
        pass
