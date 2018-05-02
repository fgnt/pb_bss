from dataclasses import dataclass

import numpy as np


@dataclass
class ComplexWatsonParameters:
    mode: np.array = None
    concentration: np.array = None


class ComplexWatson:
    """
    >>> import dc_integration.distributions.complex_watson as cw
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> scales = [
    ...     np.arange(0, 0.01, 0.001),
    ...     np.arange(0, 20, 0.01),
    ...     np.arange(0, 100, 1)
    ... ]
    >>> functions = [
    ...     cw.ComplexWatson.log_norm_low_concentration,
    ...     cw.ComplexWatson.log_norm_medium_concentration,
    ...     cw.ComplexWatson.log_norm_high_concentration
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

    log_norm = log_norm_medium_concentration

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
        """ This is twice as slow as interpolation with Tran Vu's C-code."""
        return self.spline(eigenvalues)

    def fit(self) -> ComplexWatsonParameters:
        pass
