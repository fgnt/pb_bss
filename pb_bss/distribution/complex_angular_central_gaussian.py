"""
References:
    David E. Tyler
    Statistical analysis for the angular central Gaussian distribution on the
    sphere
    1987

    N. Ito, S. Araki, T. Nakatani
    Complex angular central Gaussian mixture model for directional statistics in
    mask-based microphone array signal processing
    2016
    https://www.eurasip.org/Proceedings/Eusipco/Eusipco2016/papers/1570256519.pdf
"""

from dataclasses import dataclass

import numpy as np
from pb_bss.utils import is_broadcast_compatible
from pb_bss.distribution.utils import force_hermitian
from pb_bss.distribution.utils import _ProbabilisticModel
from pb_bss.distribution.utils import _unit_norm
from pb_bss.distribution.complex_circular_symmetric_gaussian import (
    ComplexCircularSymmetricGaussian
)


__all__ = [
    'ComplexAngularCentralGaussian',
    'ComplexAngularCentralGaussianTrainer',
    'sample_complex_angular_central_gaussian',
]


def normalize_observation(observation):
    """

    Attention: swap D and N dim

    The dimensions are swapped, because some calculations (e.g. covariance) do
    a reduction over the sample (time) dimension. Having the time dimension on
    the last axis improves the execution time.

    Args:
        observation: (..., N, D)

    Returns:
        normalized observation (..., D, N)
    """
    observation = _unit_norm(
        observation,
        axis=-1,
        eps=np.finfo(observation.dtype).tiny,
        eps_style='where',
    )
    return np.ascontiguousarray(np.swapaxes(observation, -2, -1))


def sample_complex_angular_central_gaussian(
        size,
        covariance,
):
    csg = ComplexCircularSymmetricGaussian(covariance=covariance)
    x = csg.sample(size=size)
    x /= np.linalg.norm(x, axis=-1, keepdims=True)
    return x


@dataclass
class ComplexAngularCentralGaussian(_ProbabilisticModel):
    """


    Note:
        Instead of the covariance the eigenvectors and eigenvalues are saved.
        These saves some computations, because to have a more stable covariance,
        the eigenvalues are floored.
    """
    covariance_eigenvectors: np.array = None  # (..., D, D)
    covariance_eigenvalues: np.array = None  # (..., D)

    @classmethod
    def from_covariance(
            cls,
            covariance,
            eigenvalue_floor=0.,
            covariance_norm='eigenvalue',
    ):
        if covariance_norm == 'trace':
            cov_trace = np.einsum('...dd', covariance)[..., None, None]
            covariance /= np.maximum(cov_trace, np.finfo(cov_trace.dtype).tiny)
        else:
            assert covariance_norm in ['eigenvalue', False]

        try:
            eigenvals, eigenvecs = np.linalg.eigh(covariance)
        except np.linalg.LinAlgError:
            # ToDo: figure out when this happen and why eig may work.
            # It is likely that eig is more stable than eigh.
            try:
                eigenvals, eigenvecs = np.linalg.eig(covariance)
            except np.linalg.LinAlgError:
                if eigenvalue_floor == 0:
                    raise RuntimeError(
                        'When you set the eigenvalue_floor to zero it can '
                        'happen that the eigenvalues get zero and the '
                        'reciprocal eigenvalue that is used in '
                        f'{cls.__name__}._log_pdf gets infinity.'
                    )
                else:
                    raise
        eigenvals = eigenvals.real
        if covariance_norm == 'eigenvalue':
            # The scale of the eigenvals does not matter.
            eigenvals = eigenvals / np.maximum(
                np.amax(eigenvals, axis=-1, keepdims=True),
                np.finfo(eigenvals.dtype).tiny,
            )
            eigenvals = np.maximum(
                eigenvals,
                eigenvalue_floor,
            )
        else:
            eigenvals = np.maximum(
                eigenvals,
                np.amax(eigenvals, axis=-1, keepdims=True) * eigenvalue_floor,
            )

        return cls(
            covariance_eigenvalues=eigenvals,
            covariance_eigenvectors=eigenvecs,
        )

    def sample(self, size):
        return sample_complex_angular_central_gaussian(
            size=size,
            covariance=self.covariance,
        )

    @property
    def covariance(self):
        return np.einsum(
            '...wx,...x,...zx->...wz',
            self.covariance_eigenvectors,
            self.covariance_eigenvalues,
            self.covariance_eigenvectors.conj(),
            optimize='greedy',
        )

    @property
    def log_determinant(self):
        return np.sum(np.log(self.covariance_eigenvalues), axis=-1)

    def log_pdf(self, y):
        """

        Args:
            y: Shape (..., N, D)

        Returns:

        """
        y = normalize_observation(y)  # swap D and N dim
        log_pdf, _ = self._log_pdf(y)
        return log_pdf

    def _log_pdf(self, y):
        """Gets used by. e.g. the cACGMM.
        TODO: quadratic_form might be useful by itself

        Note: y shape is (..., D, N) and not (..., N, D) like in log_pdf

        Args:
            y: Normalized observations with shape (..., D, N).
        Returns: Affiliations with shape (..., K, N) and quadratic format
            with the same shape.

        """

        *independent, D, T = y.shape

        assert is_broadcast_compatible(
            [*independent, D, D], self.covariance_eigenvectors.shape
        ), (y.shape, self.covariance_eigenvectors.shape)

        quadratic_form = np.maximum(
            np.abs(
                np.einsum(
                    # '...dt,...kde,...ke,...kge,...gt->...kt',
                    '...dt,...de,...e,...ge,...gt->...t',
                    y.conj(),
                    self.covariance_eigenvectors,
                    1 / self.covariance_eigenvalues,
                    self.covariance_eigenvectors.conj(),
                    y,
                    optimize='optimal',
                )
            ),
            np.finfo(y.dtype).tiny,
        )
        log_pdf = -D * np.log(quadratic_form)
        log_pdf -= self.log_determinant[..., None]

        return log_pdf, quadratic_form


class ComplexAngularCentralGaussianTrainer:
    def fit(
            self,
            y,
            saliency=None,
            hermitize=True,
            covariance_norm='eigenvalue',
            eigenvalue_floor=1e-10,
            iterations=10,
    ):
        """

        Args:
            y: Should be normalized to unit norm. We normalize it anyway again.
               Shape (..., D, N), e.g. (1, D, N) for mixture models
            saliency: Shape (..., N), e.g. (K, N) for mixture models
            hermitize:
            eigenvalue_floor:
            iterations:

        Returns:

        """
        *independent, N, D = y.shape
        assert np.iscomplexobj(y), y.dtype
        assert y.shape[-1] > 1
        y = normalize_observation(y)  # swap D and N dim

        if saliency is None:
            quadratic_form = np.ones(*independent, N)
        else:
            raise NotImplementedError

        assert iterations > 0, iterations
        for _ in range(iterations):
            model = self._fit(
                y=y,
                saliency=saliency,
                quadratic_form=quadratic_form,
                hermitize=hermitize,
                covariance_norm=covariance_norm,
                eigenvalue_floor=eigenvalue_floor,
            )
            _, quadratic_form = model._log_pdf(y)

        return model

    def _fit(
            self,
            y,
            saliency,
            quadratic_form,
            hermitize=True,
            covariance_norm='eigenvalue',
            eigenvalue_floor=1e-10,
    ) -> ComplexAngularCentralGaussian:
        """Single step of the fit function. In general, needs iterations.

        Note: y shape is (..., D, N) and not (..., N, D) like in fit

        Args:
            y:  Assumed to have unit length.
                Shape (..., D, N), e.g. (1, D, N) for mixture models
            saliency: Shape (..., N), e.g. (K, N) for mixture models
            quadratic_form: (..., N), e.g. (K, N) for mixture models
            hermitize:
            eigenvalue_floor:

        Returns:

        """
        assert np.iscomplexobj(y), y.dtype

        assert is_broadcast_compatible(
            y.shape[:-2], quadratic_form.shape[:-1]
        ), (y.shape, quadratic_form.shape)

        D = y.shape[-2]
        *independent, N = quadratic_form.shape

        if saliency is None:
            saliency = 1
            denominator = N
        else:
            assert y.ndim == saliency.ndim + 1, (y.shape, saliency.ndim)
            denominator = np.einsum('...n->...', saliency)[..., None, None]

        covariance = D * np.einsum(
            '...n,...dn,...Dn->...dD',
            (saliency / quadratic_form),
            y,
            y.conj(),
            optimize='greedy',
        )
        covariance /= denominator
        assert covariance.shape == (*independent, D, D), covariance.shape

        if hermitize:
            covariance = force_hermitian(covariance)

        return ComplexAngularCentralGaussian.from_covariance(
            covariance,
            eigenvalue_floor=eigenvalue_floor,
            covariance_norm=covariance_norm,
        )
