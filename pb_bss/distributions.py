"""Contains all distributions which are possibly used for BSS."""
import math
import warnings
import numpy as np
from scipy.interpolate import interp1d
from scipy.special import hyp1f1
from scipy.special import ive
import matplotlib.pyplot as plt

from pb_bss.utils import reshape
from pb_bss.utils import get_pca
from pb_bss.utils import get_power_spectral_density_matrix
from pb_bss.utils import get_stft_center_frequencies
from pb_bss.utils import deprecated

try:
    from nt.visualization import plot, facet_grid, context_manager
except ImportError:
    warnings.warn('Visual debugging not possible.')


DEPRECATION_MESSAGE = (
        'Most functionality moved to the dc_integration.distribution module.'
)

@deprecated(DEPRECATION_MESSAGE)
def _unit_norm(signal):
    """Unit normalization.

    Args:
        signal: STFT signal with shape (..., T, D).
    Returns:
        Normalized STFT signal with same shape.
    """
    return signal / (np.linalg.norm(signal, axis=-1, keepdims=True) + 1e-4)


@deprecated(DEPRECATION_MESSAGE)
def _phase_norm(signal, reference_channel=0):
    """Unit normalization.

    Args:
        signal: STFT signal with shape (..., T, D).
    Returns:
        Normalized STFT signal with same shape.
    """
    angles = np.angle(signal[..., [reference_channel]])
    return signal * np.exp(-1j * angles)


@deprecated(DEPRECATION_MESSAGE)
def _frequency_norm(
        signal,
        max_sensor_distance=None, shrink_factor=1.2,
        fft_size=1024, sample_rate=16000, sound_velocity=343
):
    """Unit normalization.

    Aside from this function, the whole class is frequency independent.
    This function is not really tested, since the use case vanished.

    Args:
        signal: STFT signal with shape (F, T, D).
        max_sensor_distance: Distance in meter.
        shrink_factor: Heuristic shrink factor to move further away from
            the wrapping boarder.
        fft_size:
        sample_rate: In hertz.
        sound_velocity: Speed in meter per second.
    Returns:
        Normalized STFT signal with same shape.
    """
    frequency = get_stft_center_frequencies(fft_size, sample_rate)
    F, _, _ = signal.shape
    assert len(frequency) == F
    norm_factor = sound_velocity / (
        2 * frequency * shrink_factor * max_sensor_distance
    )

    # Norm factor can become NaN when one center frequency is zero.
    norm_factor = np.nan_to_num(norm_factor)
    if norm_factor[-1] < 1:
        raise ValueError(
            'Distance between the sensors too high: {:.2} > {:.2}'.format(
                max_sensor_distance, sound_velocity / (2 * frequency[-1])
            )
        )

    norm_factor = norm_factor[:, None, None]
    signal = np.abs(signal) * np.exp(1j * np.angle(signal) * norm_factor)


class ComplexWatson:
    """
    >>> import pb_bss.distributions.complex_watson as cw
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
    @deprecated(DEPRECATION_MESSAGE)
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
    @deprecated(DEPRECATION_MESSAGE)
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
    @deprecated(DEPRECATION_MESSAGE)
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
    @deprecated(DEPRECATION_MESSAGE)
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
    @deprecated(DEPRECATION_MESSAGE)
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

    @deprecated(DEPRECATION_MESSAGE)
    def __init__(self, dimension, max_concentration=100, spline_markers=100):
        self.dimension = dimension
        self.max_concentration = max_concentration
        self.spline_markers = spline_markers
        self.spline = self._get_spline()

    @deprecated(DEPRECATION_MESSAGE)
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

    @deprecated(DEPRECATION_MESSAGE)
    def hypergeometric_ratio_inverse(self, eigenvalues):
        """ This is twice as slow as interpolation with Tran Vu's C-code."""
        return self.spline(eigenvalues)


class ComplexWatsonMixtureModel:
    """Collects all functions related to the cWMM."""
    unit_norm = staticmethod(_unit_norm)
    phase_norm = staticmethod(_phase_norm)
    frequency_norm = staticmethod(_frequency_norm)

    @deprecated(DEPRECATION_MESSAGE)
    def __init__(self, pi=None, W=None, kappa=None):
        """Initializes empty instance variables.

        Args:
            pi: Mixture weights with shape (..., K).
            W: Mode vectors with shape (..., K, D).
            kappa: Concentration parameters with shape (..., K).
        """
        self.pi = np.empty((), dtype=np.float) if pi is None else pi
        self.W = np.empty((), dtype=np.float) if W is None else W
        self.kappa = np.empty((), dtype=np.float) if kappa is None else kappa

    @deprecated(DEPRECATION_MESSAGE)
    def fit(
            self, Y_normalized, initialization,
            iterations=100, max_concentration=100
    ):
        """ EM for CWMMs with any number of independent dimensions.

        Does not support sequence lengths.
        Can later be extended to accept more initializations, but for now
        only accepts affiliations (masks) as initialization.

        Args:
            Y_normalized: Mix with shape (..., T, D).
            initialization: Shape (..., K, T)
            iterations: Most of the time 10 iterations are acceptable.
            max_concentration: For numerical stability reasons.
        """
        Y_normalized_for_pdf = np.copy(Y_normalized, 'C')
        Y_normalized_for_psd = np.copy(np.swapaxes(Y_normalized, -2, -1), 'C')

        D = Y_normalized.shape[-2]
        cw = ComplexWatson(D, max_concentration=max_concentration)

        affiliations = np.copy(initialization)

        for i in range(iterations):
            # E step
            if i > 0:
                affiliations = self.predict(Y_normalized_for_pdf)

            # M step
            self.pi = np.mean(affiliations, axis=-1)
            Phi = get_power_spectral_density_matrix(
                Y_normalized_for_psd, np.copy(affiliations, 'C'),
                sensor_dim=-2, source_dim=-2, time_dim=-1
            )

            self.W, eigenvalues = get_pca(Phi)
            self.kappa = cw.hypergeometric_ratio_inverse(eigenvalues)

    @deprecated(DEPRECATION_MESSAGE)
    def predict(self, Y_normalized):
        """Predict class affiliation posteriors from given model.

        Args:
            Y_normalized: Mix with shape (..., T, D).
        Returns: Affiliations with shape (..., K, T).
        """
        affiliations = self.pi[..., None] * ComplexWatson.pdf(
            Y_normalized[..., None, :, :],
            np.copy(self.W[..., None, :], 'C'),
            self.kappa[..., None]
        )
        affiliations /= np.sum(affiliations, axis=-2, keepdims=True)
        return affiliations


class ComplexGaussianMixtureModel:
    """TV-cGMM.

    Higuchi, T.; Yoshioka, T. & Nakatani, T.
    Optimization of Speech Enhancement Front-End with Speech Recognition-Level
    Criterion
    Interspeech 2016, 2016, 3808-3812

    Original paper did not use mixture weights. In contrast to the original
    paper we make use of Eigen decomposition to avoid direct computation of
    determinant and inverse.

    This algorithm does not work well with too few channels.
    At least 4 channels are necessary for proper mask results.
    """

    @deprecated(DEPRECATION_MESSAGE)
    def __init__(
            self, use_mixture_weights=False,
            eps=1e-10, visual_debug=False
    ):
        """Initializes empty instance variables.

        Shapes:
            pi: Mixture weights with shape (..., K).
            covariance: Covariance matrices with shape (..., K, D, D).
        """
        self.use_mixture_weights = use_mixture_weights
        self.eps = eps
        self.visual_debug = visual_debug
        self.covariance = np.empty((), dtype=np.complex)
        self.precision = np.empty((), dtype=np.complex)
        self.determinant = np.empty((), dtype=np.float)

        if self.use_mixture_weights:
            self.pi = np.empty((), dtype=np.float)

    @deprecated(DEPRECATION_MESSAGE)
    def fit(
            self, Y, initialization, iterations=100,
            hermitize=True, trace_norm=True, eigenvalue_floor=1e-10,
            inverse='inv'
    ):
        """ EM for cGMM with any number of independent dimensions.

        Does not support sequence lengths.
        Can later be extended to accept more initializations.

        Args:
            Y: Mix with shape (..., T, D).
            iterations:
            initialization: Shape (..., K, T).
        """
        Y_for_psd = np.copy(np.swapaxes(Y, -2, -1), 'C')
        Y_for_pdf = np.copy(Y, 'C')
        D = Y_for_pdf.shape[-1]

        affiliations = np.copy(initialization)
        power = np.ones_like(affiliations)

        for i in range(iterations):
            # E step
            if i > 0:
                # Equation 10
                affiliations, power = self._predict(Y_for_pdf)

            # M step
            if self.use_mixture_weights:
                self.pi = np.mean(affiliations, axis=-1)

            # Equation 6
            self.covariance = get_power_spectral_density_matrix(
                Y_for_psd,
                np.copy(np.clip(affiliations, self.eps, 1 - self.eps) / power,
                        'C'),
                sensor_dim=-2, source_dim=-2, time_dim=-1
            )

            if hermitize:
                self.covariance = (
                                      self.covariance
                                      + np.swapaxes(self.covariance.conj(), -1,
                                                    -2)
                                  ) / 2

            if trace_norm:
                self.covariance /= np.einsum(
                    '...dd', self.covariance
                )[..., None, None]

            # Deconstructs covariance matrix and constrains eigenvalues
            eigenvals, eigenvecs = np.linalg.eigh(self.covariance)
            eigenvals = eigenvals.real
            eigenvals = np.maximum(
                eigenvals,
                np.max(eigenvals, axis=-1, keepdims=True) * eigenvalue_floor
            )
            diagonal = np.einsum('de,fkd->fkde', np.eye(D), eigenvals)
            self.covariance = np.einsum(
                'fkwx,fkxy,fkzy->fkwz', eigenvecs, diagonal, eigenvecs.conj()
            )
            self.determinant = np.prod(eigenvals, axis=-1)
            inverse_diagonal = np.einsum('de,fkd->fkde', np.eye(D),
                                         1 / eigenvals)
            self.precision = np.einsum(
                'fkwx,fkxy,fkzy->fkwz', eigenvecs, inverse_diagonal,
                eigenvecs.conj()
            )

            if self.visual_debug:
                with context_manager(figure_size=(24, 3)):
                    plt.plot(np.log10(
                        np.max(eigenvals, axis=-1)
                        / np.min(eigenvals, axis=-1)
                    ))
                    plt.xlabel('frequency bin')
                    plt.ylabel('eigenvalue spread')
                    plt.show()
                with context_manager(figure_size=(24, 3)):
                    plt.plot(self.pi)
                    plt.show()

    @deprecated(DEPRECATION_MESSAGE)
    def _predict(self, Y, inverse='inv'):
        D = Y.shape[-1]

        if inverse == 'inv':
            precision = np.linalg.inv(self.covariance)
            power = np.abs(
                np.einsum('...td,...kde,...te->...kt', Y.conj(), precision, Y)
            ) / D + self.eps
        elif inverse == 'solve':
            Y_with_inverse_covariance = np.linalg.solve(
                self.covariance[..., None, :, :],
                Y[..., None, :, :]
            )
            power = np.einsum(
                '...td,...ktd->...kt', Y.conj(), Y_with_inverse_covariance
            ).real / D + self.eps
        elif inverse == 'eig':
            power = np.abs(
                np.einsum('...td,...kde,...te->...kt', Y.conj(), self.precision,
                          Y)
            ) / D + self.eps

        affiliations = np.exp(
            -np.log(self.determinant)[..., None] - D * np.log(power))

        if self.use_mixture_weights:
            affiliations *= self.pi[..., None]

        affiliations /= np.sum(affiliations, axis=-2, keepdims=True) + self.eps

        if self.visual_debug:
            _plot_affiliations(affiliations)

        return affiliations, power

    @deprecated(DEPRECATION_MESSAGE)
    def predict(self, Y, inverse='inv'):
        """Predict class affiliation posteriors from given model.

        Args:
            Y: Mix with shape (..., T, D).
        Returns: Affiliations with shape (..., K, T).
        """
        return self._predict(Y, inverse=inverse)[0]


class VonMisesFisher:
    """
    All inputs should be of type ndarray.

    Expected shapes:
    x: (D, N) SOLL (..., D)
    mu: (K, D) SOLL (..., D)
    kappa: (K,) SOLL (...)
    """

    @classmethod
    @deprecated(DEPRECATION_MESSAGE)
    def norm(cls, kappa, D):
        return np.exp(cls.log_norm(kappa, D))

    @classmethod
    @deprecated(DEPRECATION_MESSAGE)
    def pdf(cls, x, mu, kappa):
        return np.exp(cls.log_pdf(x, mu, kappa))

    @classmethod
    @deprecated(DEPRECATION_MESSAGE)
    def log_norm(cls, kappa, D):
        kappa = np.clip(kappa, 1e-10, np.inf)
        return (
            (D / 2) * np.log(2 * np.pi) + np.log(ive(D / 2 - 1, kappa))
            + np.abs(kappa) - (D / 2 - 1) * np.log(kappa)
        )

    @classmethod
    @deprecated(DEPRECATION_MESSAGE)
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
    @deprecated(DEPRECATION_MESSAGE)
    def fit(
            cls, x, weights=None,
            min_concentration=0, max_concentration=500
    ):
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


class ComplexAngularCentralGaussianMixtureModel:
    """Ito 2016."""
    unit_norm = staticmethod(_unit_norm)

    @deprecated(DEPRECATION_MESSAGE)
    def __init__(self, eps=1e-10, visual_debug=False):
        self.eps = eps
        self.visual_debug = visual_debug
        self.covariance = np.empty((), dtype=np.complex)
        self.precision = np.empty((), dtype=np.complex)
        self.determinant = np.empty((), dtype=np.float)
        self.pi = np.empty((), dtype=np.float)

    @deprecated(DEPRECATION_MESSAGE)
    def fit(
            self, Y, initialization, iterations=100,
            hermitize=True, trace_norm=True, eigenvalue_floor=1e-10
    ):
        """Fit a cACGMM.

        Args:
            Y: Normalized observations with shape (..., T, D).
            iterations:
            initialization: Shape (..., K, T).
        """
        F, T, D = Y.shape
        K = initialization.shape[-2]
        Y_for_psd = np.copy(np.swapaxes(Y, -2, -1), 'C')[..., None, :, :]
        Y_for_pdf = np.copy(Y, 'C')
        D = Y_for_pdf.shape[-1]

        affiliations = np.copy(initialization)
        quadratic_form = np.ones_like(affiliations)

        for i in range(iterations):
            # E step
            if i > 0:
                # Equation 12
                affiliations, quadratic_form = self._predict(Y_for_pdf)

            self.pi = np.mean(affiliations, axis=-1)
            assert self.pi.shape == (F, K), self.pi.shape

            mask = affiliations[..., None, :]
            assert mask.shape == (F, K, 1, T), mask.shape
            self.covariance = D * np.einsum(
                '...dt,...et->...de',
                (mask / quadratic_form[..., None, :]) * Y_for_psd,
                Y_for_psd.conj()
            )
            normalization = np.sum(mask, axis=-1, keepdims=True)
            self.covariance /= normalization
            assert self.covariance.shape == (F, K, D, D), self.covariance.shape

            if hermitize:
                self.covariance = (
                                      self.covariance
                                      + np.swapaxes(self.covariance.conj(), -1,
                                                    -2)
                                  ) / 2

            if trace_norm:
                self.covariance /= np.einsum(
                    '...dd', self.covariance
                )[..., None, None]

            eigenvals, eigenvecs = np.linalg.eigh(self.covariance)
            eigenvals = eigenvals.real
            eigenvals = np.maximum(
                eigenvals,
                np.max(eigenvals, axis=-1, keepdims=True) * eigenvalue_floor
            )
            diagonal = np.einsum('de,fkd->fkde', np.eye(D), eigenvals)
            self.covariance = np.einsum(
                'fkwx,fkxy,fkzy->fkwz', eigenvecs, diagonal, eigenvecs.conj()
            )
            self.determinant = np.prod(eigenvals, axis=-1)
            inverse_diagonal = np.einsum('de,fkd->fkde', np.eye(D),
                                         1 / eigenvals)
            self.precision = np.einsum(
                'fkwx,fkxy,fkzy->fkwz', eigenvecs, inverse_diagonal,
                eigenvecs.conj()
            )

            if self.visual_debug:
                with context_manager(figure_size=(24, 3)):
                    plt.plot(np.log10(
                        np.max(eigenvals, axis=-1)
                        / np.min(eigenvals, axis=-1)
                    ))
                    plt.xlabel('frequency bin')
                    plt.ylabel('eigenvalue spread')
                    plt.show()
                with context_manager(figure_size=(24, 3)):
                    plt.plot(self.pi)
                    plt.show()

    @deprecated(DEPRECATION_MESSAGE)
    def _predict(self, Y):
        """Predict class affiliation posteriors from given model.

        Args:
            Y: Normalized observations with shape (..., T, D).
        Returns: Affiliations with shape (..., K, T) and quadratic format
            with the same shape.
        """
        F, T, D = Y.shape
        K = self.pi.shape[-1]
        D = Y.shape[-1]

        quadratic_form = np.abs(
            np.einsum('...td,...kde,...te->...kt', Y.conj(), self.precision, Y)
        ) + self.eps
        assert quadratic_form.shape == (F, K, T), quadratic_form.shape

        affiliations = - D * np.log(quadratic_form)
        affiliations -= np.log(self.determinant)[..., None]
        affiliations = np.exp(affiliations)
        affiliations *= self.pi[..., None]
        affiliations /= np.sum(affiliations, axis=-2, keepdims=True) + self.eps

        affiliations = np.clip(affiliations, self.eps, 1 - self.eps)

        if self.visual_debug:
            _plot_affiliations(affiliations)

        return affiliations, quadratic_form

    @deprecated(DEPRECATION_MESSAGE)
    def predict(self, Y):
        """Predict class affiliation posteriors from given model.

        Args:
            Y: Normalized observations with shape (..., T, D).
        Returns: Affiliations with shape (..., K, T).
        """
        return self._predict(Y)[0]


class VonMisesFisherMixtureModel:
    """The vMFMM can be used to cluster the embeddings."""
    unit_norm = staticmethod(_unit_norm)

    @deprecated(DEPRECATION_MESSAGE)
    def __init__(self, eps=1e-10, visual_debug=False):
        self.eps = eps
        self.visual_debug = visual_debug
        self.mu = None
        self.kappa = None
        self.pi = None

    @deprecated(DEPRECATION_MESSAGE)
    def fit(
            self, x, initialization,
            iterations=100, min_concentration=0, max_concentration=500
    ):
        """ EM for vMFMMs with any number of independent dimensions.

        Args:
            x: Observations with shape (N, D).
                Observations are expected to are unit norm normalized.
            initialization: Shape (..., K, N)
        """
        affiliations = np.copy(initialization)

        for i in range(iterations):
            # E-step
            if i > 0:
                affiliations = self.predict(x)

            affiliations = np.clip(affiliations, self.eps, 1 - self.eps)

            # M-step
            self.mu, self.kappa = VonMisesFisher.fit(
                x[..., None, :, :],
                affiliations,
                min_concentration=min_concentration,
                max_concentration=max_concentration
            )
            self.pi = np.mean(affiliations, axis=-1)

            if self.visual_debug:
                print('self.pi', self.pi)
                print('self.kappa', self.kappa)

        return affiliations

    @deprecated(DEPRECATION_MESSAGE)
    def predict(self, x):
        """Predict class affiliation posteriors from given model.

        Args:
            x: Observations with shape (..., N, D).
                Observations are expected to are unit norm normalized.
        Returns: Affiliations with shape (..., K, T).
        """
        affiliations = VonMisesFisher.pdf(
            x[..., None, :, :],
            self.mu[..., None, :],
            self.kappa[..., None]
        ) * self.pi[..., :, None]
        affiliations /= np.sum(affiliations, axis=-2, keepdims=True)

        if self.visual_debug:
            _plot_affiliations(np.reshape(affiliations[0], (-1, 1, 257)).T)

        return affiliations


class VonMisesFisherComplexWatsonMixtureModel:
    """Hybrid model."""
    unit_norm = staticmethod(_unit_norm)
    phase_norm = staticmethod(_phase_norm)
    frequency_norm = staticmethod(_frequency_norm)

    @deprecated(DEPRECATION_MESSAGE)
    def __init__(self, *, spatial_score, embedding_score):
        self.spatial_score = spatial_score
        self.embedding_score = embedding_score
        self.mu = np.empty((), dtype=np.float)
        self.kappa_vmf = np.empty((), dtype=np.float)
        self.pi = np.empty((), dtype=np.float)
        self.W = np.empty((), dtype=np.float)
        self.kappa_cw = np.empty((), dtype=np.float)

    @deprecated(DEPRECATION_MESSAGE)
    def fit(
            self, Y_normalized, embedding, initialization, iterations=100,
            max_concentration_cw=100, max_concentration_vmf=500
    ):
        """

        Args:
            Y_normalized: Mix with shape (..., T, D).
            embedding: Embedding from Deep Clustering with shape (..., T, E).
            initialization: Shape (..., K, T)
            iterations:
            max_concentration_cw:
            max_concentration_vmf:

        Returns:

        """
        Y_normalized_for_psd = np.copy(np.swapaxes(Y_normalized, -2, -1), 'C')
        Y_normalized_for_pdf = np.copy(Y_normalized, 'C')
        affiliations = np.copy(initialization)
        D = Y_normalized.shape[-2]
        cw = ComplexWatson(D, max_concentration=max_concentration_cw)

        for i in range(iterations):
            # E step
            if i > 0:
                affiliations = self.predict(Y_normalized_for_pdf, embedding)

            # M step
            self.pi = affiliations.mean(axis=-1)
            Phi = get_power_spectral_density_matrix(
                Y_normalized_for_psd, np.copy(affiliations, 'C'),
                sensor_dim=-2, source_dim=-2, time_dim=-1
            )
            self.W, eigenvalues = get_pca(Phi)
            self.kappa_cw = cw.hypergeometric_ratio_inverse(eigenvalues)
            self.mu, self.kappa_vmf = VonMisesFisher.fit(
                embedding, reshape(affiliations, 'fkt->k,t*f'),
                max_concentration=max_concentration_vmf
            )

    @deprecated(DEPRECATION_MESSAGE)
    def predict(self, Y_normalized, embedding):
        """Predict class affiliation posteriors from given model.

        Args:
            Y_normalized: Mix with shape (..., T, D).
            embedding: Embedding from Deep Clustering with shape (F*T, E).
        Returns: Affiliations with shape (..., K, T).
        """
        K = self.pi.shape[-1]
        T = Y_normalized.shape[-2]
        F = Y_normalized.shape[-3]
        affiliations = ComplexWatson.pdf(
            Y_normalized[..., None, :, :],
            np.copy(self.W[..., None, :], 'C'),
            self.kappa_cw[..., None]
        ) ** self.spatial_score
        affiliations *= np.reshape(
            VonMisesFisher.pdf(
                embedding[..., None, :, :],
                self.mu[..., None, :],
                self.kappa_vmf[..., None]
            ),
            (K, T, F)
        ).transpose(2, 0, 1) ** self.embedding_score
        affiliations *= self.pi[..., None]
        affiliations /= np.sum(affiliations, axis=-2, keepdims=True)
        return affiliations


class VonMisesFisherComplexGaussianMixtureModel:
    """Hybrid model."""
    unit_norm = staticmethod(_unit_norm)

    @deprecated(DEPRECATION_MESSAGE)
    def __init__(
            self, *, spatial_score, embedding_score,
            eps=1e-10, visual_debug=False
    ):
        self.spatial_score = spatial_score
        self.embedding_score = embedding_score
        self.eps = eps
        self.visual_debug = visual_debug
        self.mu = np.empty((), dtype=np.float)
        self.kappa_vmf = np.empty((), dtype=np.float)
        self.pi = np.empty((), dtype=np.float)
        self.covariance = np.empty((), dtype=np.float)
        self.precision = np.empty((), dtype=np.float)
        self.determinant = np.empty((), dtype=np.float)

    @deprecated(DEPRECATION_MESSAGE)
    def fit(
            self, Y, embedding, initialization, iterations=100,
            min_concentration_vmf=0, max_concentration_vmf=500,
            hermitize=True, trace_norm=True, eigenvalue_floor=1e-10,
            inverse='inv'
    ):
        """

        Args:
            Y: Mix with shape (F, D, T).
            embedding: Embedding from Deep Clustering with shape (F*T, E).
            initialization: Shape (F, K, T)
            iterations: Most of the time 10 iterations are acceptable.
            min_concentration_vmf: For numerical stability reasons.
            max_concentration_vmf: For numerical stability reasons.

        Returns:
        """
        D = Y.shape[-1]
        Y_for_psd = np.copy(np.swapaxes(Y, -2, -1), 'C')
        Y_for_pdf = np.copy(Y, 'C')
        embedding = np.copy(np.swapaxes(embedding, -2, -1), 'C')

        # F, K, T = initialization.shape[-3:]
        affiliations = np.copy(initialization)
        power = np.ones_like(affiliations)

        for i in range(iterations):
            # E step
            if i > 0:
                affiliations, power = self._predict(Y_for_pdf, embedding.T,
                                                    inverse=inverse)

            # M step
            self.pi = affiliations.mean(axis=-1)

            self.covariance = get_power_spectral_density_matrix(
                Y_for_psd,
                np.copy(np.clip(affiliations, self.eps, 1 - self.eps) / power,
                        'C'),
                sensor_dim=-2, source_dim=-2, time_dim=-1
            )

            if hermitize:
                self.covariance = (
                                      self.covariance
                                      + np.swapaxes(self.covariance.conj(), -1,
                                                    -2)
                                  ) / 2

            if trace_norm:
                self.covariance /= np.einsum(
                    '...dd', self.covariance
                )[..., None, None]

            # Deconstructs covariance matrix and constrains eigenvalues
            eigenvals, eigenvecs = np.linalg.eigh(self.covariance)
            eigenvals = eigenvals.real
            eigenvals = np.maximum(
                eigenvals,
                np.max(eigenvals, axis=-1, keepdims=True) * eigenvalue_floor
            )
            diagonal = np.einsum('de,fkd->fkde', np.eye(D), eigenvals)
            self.covariance = np.einsum(
                'fkwx,fkxy,fkzy->fkwz', eigenvecs, diagonal, eigenvecs.conj()
            )
            self.determinant = np.prod(eigenvals, axis=-1)
            inverse_diagonal = np.einsum('de,fkd->fkde', np.eye(D),
                                         1 / eigenvals)
            self.precision = np.einsum(
                'fkwx,fkxy,fkzy->fkwz', eigenvecs, inverse_diagonal,
                eigenvecs.conj()
            )

            if self.visual_debug:
                with context_manager(figure_size=(24, 3)):
                    plt.plot(np.log10(
                        np.max(eigenvals, axis=-1)
                        / np.min(eigenvals, axis=-1)
                    ))
                    plt.xlabel('frequency bin')
                    plt.ylabel('eigenvalue spread')
                    plt.show()

            self.mu, self.kappa_vmf = VonMisesFisher.fit(
                embedding.T,
                np.clip(reshape(affiliations, 'fkt->k,t*f'), self.eps,
                        1 - self.eps),
                min_concentration=min_concentration_vmf,
                max_concentration=max_concentration_vmf
            )

    @deprecated(DEPRECATION_MESSAGE)
    def _predict(self, Y, embedding, inverse='inv'):
        D = Y.shape[-1]
        K = self.covariance.shape[-3]
        T = Y.shape[-2]
        F = Y.shape[-3]

        if inverse == 'inv':
            precision = np.linalg.inv(self.covariance)
            power = np.abs(
                np.einsum('...td,...kde,...te->...kt', Y.conj(), precision, Y)
            ) / D + self.eps
        elif inverse == 'solve':
            Y_with_inverse_covariance = np.linalg.solve(
                self.covariance[..., None, :, :],
                Y[..., None, :, :]
            )
            power = np.einsum(
                '...td,...ktd->...kt', Y.conj(), Y_with_inverse_covariance
            ).real / D + self.eps
        elif inverse == 'eig':
            power = np.abs(
                np.einsum('...td,...kde,...te->...kt', Y.conj(), self.precision,
                          Y)
            ) / D + self.eps

        spatial = np.exp(
            -np.log(self.determinant)[..., None] - D * np.log(power))

        emb = np.reshape(
            VonMisesFisher.pdf(
                embedding[..., None, :, :],
                self.mu[..., None, :],
                self.kappa_vmf[..., None]
            ),
            (K, T, F)
        ).transpose(2, 0, 1)

        affiliations = spatial ** self.spatial_score
        affiliations *= emb ** self.embedding_score
        affiliations *= self.pi[..., None]
        affiliations /= np.sum(affiliations, axis=-2, keepdims=True) + self.eps

        if self.visual_debug:
            # Normalization only necessary for visualization
            spatial /= np.sum(spatial, axis=-2, keepdims=True) + self.eps
            emb /= np.sum(emb, axis=-2, keepdims=True) + self.eps
            _plot_affiliations(spatial, emb, affiliations)

        return affiliations, power

    @deprecated(DEPRECATION_MESSAGE)
    def predict(self, Y, embedding, inverse='inv'):
        """Predict class affiliation posteriors from given model.

        Args:
            Y: Mix with shape (..., T, D).
            embedding: Embedding from Deep Clustering with shape (F*T, E).
        Returns: Affiliations with shape (..., K, T).
        """
        return self._predict(Y, embedding, inverse=inverse)[0]


class VonMisesFisherComplexAngularCentralGaussianMixtureModel:
    """Hybrid model."""
    unit_norm = staticmethod(_unit_norm)

    @deprecated(DEPRECATION_MESSAGE)
    def __init__(
            self, *, spatial_score, embedding_score,
            eps=1e-10, visual_debug=False
    ):
        self.spatial_score = spatial_score
        self.embedding_score = embedding_score
        self.eps = eps
        self.visual_debug = visual_debug
        self.mu = np.empty((), dtype=np.float)
        self.kappa_vmf = np.empty((), dtype=np.float)
        self.covariance = np.empty((), dtype=np.complex)
        self.precision = np.empty((), dtype=np.complex)
        self.determinant = np.empty((), dtype=np.float)
        self.pi = np.empty((), dtype=np.float)

    @deprecated(DEPRECATION_MESSAGE)
    def fit(
            self, Y, embedding, initialization, iterations=100,
            min_concentration_vmf=0, max_concentration_vmf=500,
            eigenvalue_floor=1e-10
    ):
        """Fit a vMFcACGMM.

        Args:
            Y: Mix with shape (F, D, T).
            embedding: Embedding from Deep Clustering with shape (F*T, E).
            initialization: Shape (F, K, T)
            iterations: Most of the time 10 iterations are acceptable.
            min_concentration_vmf: For numerical stability reasons.
            max_concentration_vmf: For numerical stability reasons.

        Returns:
        """
        F, T, D = Y.shape
        D = Y.shape[-1]
        Y_for_psd = np.copy(np.swapaxes(Y, -2, -1), 'C')
        Y_for_pdf = np.copy(Y, 'C')
        embedding = np.copy(np.swapaxes(embedding, -2, -1), 'C')

        # F, K, T = initialization.shape[-3:]
        affiliations = np.copy(initialization)
        quadratic_form = np.ones_like(affiliations)

        for i in range(iterations):
            # E step
            if i > 0:
                affiliations, quadratic_form = self._predict(Y_for_pdf,
                                                             embedding)

            # M step
            self.pi = affiliations.mean(axis=-1)
            assert self.pi.shape == (F, K), self.pi.shape

            mask = affiliations[..., None, :]
            assert mask.shape == (F, K, 1, T), mask.shape
            self.covariance = D * np.einsum(
                '...dt,...et->...de',
                (mask / quadratic_form[..., None, :]) * Y_for_psd,
                Y_for_psd.conj()
            )
            normalization = np.sum(mask, axis=-1, keepdims=True)
            self.covariance /= normalization
            assert self.covariance.shape == (F, K, D, D), self.covariance.shape

            # Deconstructs covariance matrix and constrains eigenvalues
            eigenvals, eigenvecs = np.linalg.eigh(self.covariance)
            eigenvals = eigenvals.real
            eigenvals = np.maximum(
                eigenvals,
                np.max(eigenvals, axis=-1, keepdims=True) * eigenvalue_floor
            )
            diagonal = np.einsum('de,fkd->fkde', np.eye(D), eigenvals)
            self.covariance = np.einsum(
                'fkwx,fkxy,fkzy->fkwz', eigenvecs, diagonal, eigenvecs.conj()
            )
            self.determinant = np.prod(eigenvals, axis=-1)
            inverse_diagonal = np.einsum('de,fkd->fkde', np.eye(D),
                                         1 / eigenvals)
            self.precision = np.einsum(
                'fkwx,fkxy,fkzy->fkwz', eigenvecs, inverse_diagonal,
                eigenvecs.conj()
            )

            if self.visual_debug:
                with context_manager(figure_size=(24, 3)):
                    plt.plot(np.log10(
                        np.max(eigenvals, axis=-1)
                        / np.min(eigenvals, axis=-1)
                    ))
                    plt.xlabel('frequency bin')
                    plt.ylabel('eigenvalue spread')
                    plt.show()

            self.mu, self.kappa_vmf = VonMisesFisher.fit(
                embedding.T,
                np.clip(reshape(affiliations, 'fkt->k,t*f'), self.eps,
                        1 - self.eps),
                min_concentration=min_concentration_vmf,
                max_concentration=max_concentration_vmf
            )

    @deprecated(DEPRECATION_MESSAGE)
    def _predict(self, Y, embedding):
        D = Y.shape[-1]
        K = self.covariance.shape[-3]
        T = Y.shape[-2]
        F = Y.shape[-3]

        quadratic_form = np.abs(
            np.einsum('...td,...kde,...te->...kt', Y.conj(), self.precision, Y)
        ) + self.eps
        assert quadratic_form.shape == (F, K, T), quadratic_form.shape

        spatial = np.exp(- D * np.log(quadratic_form))

        emb = np.reshape(
            VonMisesFisher.pdf(
                embedding[..., None, :, :],
                self.mu[..., None, :],
                self.kappa_vmf[..., None]
            ),
            (K, T, F)
        ).transpose(2, 0, 1)

        affiliations = spatial ** self.spatial_score
        affiliations *= emb ** self.embedding_score
        affiliations *= self.pi[..., None]
        affiliations /= np.sum(affiliations, axis=-2, keepdims=True) + self.eps

        if self.visual_debug:
            # Normalization only necessary for visualization
            spatial /= np.sum(spatial, axis=-2, keepdims=True) + self.eps
            emb /= np.sum(emb, axis=-2, keepdims=True) + self.eps
            _plot_affiliations(spatial, emb, affiliations)

        return affiliations, power

    @deprecated(DEPRECATION_MESSAGE)
    def predict(self, Y, embedding):
        """Predict class affiliation posteriors from given model.

        Args:
            Y: Mix with shape (..., T, D).
            embedding: Embedding from Deep Clustering with shape (F*T, E).
        Returns: Affiliations with shape (..., K, T).
        """
        return self._predict(Y, embedding)[0]


@deprecated(DEPRECATION_MESSAGE)
def _plot_condition_number(matrix):
    """Allows matrices with shape (F, K)."""
    with context_manager(figure_size=(24, 3)):
        plt.plot(np.log(np.linalg.cond(matrix)))
        plt.xlabel('frequency bin')
        plt.ylabel('log cond A')
        plt.show()


@deprecated(DEPRECATION_MESSAGE)
def _plot_affiliations(*affiliation_list):
    """Each argument must have shape (F, K, T)."""
    with context_manager():
        facet_grid(
            [x[:, 0, :].T for x in affiliation_list],
            plot.mask, colwrap=max(2, len(affiliation_list))
        )
        plt.show()
