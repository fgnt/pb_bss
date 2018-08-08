from dataclasses import dataclass

import numpy as np

from dc_integration.distribution.utils import _Parameter, force_hermitian
from dc_integration.distribution.circular_symmetric_gaussian import CircularSymmetricGaussian


def sample_complex_angular_central_gaussian(
        size,
        covariance,
):
    csg = CircularSymmetricGaussian(covariance=covariance)
    x = csg.sample(size=size)
    x /= np.linalg.norm(x, axis=-1, keepdims=True)
    return x

@dataclass
class ComplexAngularCentralGaussianParameters(_Parameter):
    # covariance: np.array = None
    # precision: np.array = None
    covariance_eigenvectors: np.array = None
    covariance_eigenvalues: np.array = None
    determinant: np.array = None
    log_determinant: np.array = None

    @property
    def covariance(self):
        return np.einsum(
            '...wx,...x,...zx->...wz',
            self.covariance_eigenvectors,
            self.covariance_eigenvalues,
            self.covariance_eigenvectors.conj(),
            optimize='greedy',
        )

    def sample(self, size):
        csg = CircularSymmetricGaussian(covariance=self.covariance)
        x = csg.sample(size=size)
        x /= np.linalg.norm(x, axis=-1, keepdims=True)
        return x


class ComplexAngularCentralGaussian:

    def __init__(self, use_pinv=False):
        self.use_pinv = use_pinv

    def _fit(
            self,
            Y,
            saliency,
            quadratic_form,
            hermitize=True,
            trace_norm=True,
            eigenvalue_floor=1e-10,
            calculate_covariance=False,
    ) -> ComplexAngularCentralGaussianParameters:
        """
        Attention: Y has swapped dimensions.

        Note:
            `fit` needs an iteration for the parameter estimation, becasue
            quadratic_form depends on covariance and covariance depends on
            quadratic_form.
            Until now nobody need fit, therefore it is not implemented.

        Args:
            Y: Complex observation with shape (..., D, T)
            saliency: Weight for each observation (..., T)

        In case of mixture model:
            Y: Complex observation with shape (..., 1, D, T)
            saliency: Weight for each observation (..., K, T)
        Returns:

        ToDo: Merge ComplexGaussian with ComplexAngularCentralGaussian.
              Both have the same _fit

        """

        # def finite_check(arr):
        #     return np.all(np.isfinite(arr))

        # assert finite_check(saliency), saliency
        # assert finite_check(quadratic_form), quadratic_form
        # assert finite_check(Y), Y

        assert Y.ndim == saliency.ndim + 1, (Y.shape, saliency.ndim)
        *independent, D, T = Y.shape
        independent = list(independent)

        # Special case for mixture model
        independent[-1] = saliency.shape[-2]
        # K = saliency.shape[-2]

        params = ComplexAngularCentralGaussianParameters()

        mask = saliency[..., None, :]
        assert mask.shape == (*independent, 1, T), (mask.shape, (*independent, 1, T))

        # params.covariance = D * np.einsum(
        #     '...dt,...et->...de',
        #     (saliency / quadratic_form)[..., None, :] * Y,
        #     Y.conj()
        # )
        covariance = D * np.einsum(
            '...t,...dt,...et->...de',
            # '....dt,...et->...de',
            (saliency / quadratic_form),
            Y,
            Y.conj(),
            optimize='greedy',
        )
        # assert False
        # assert finite_check(params.covariance), params.covariance

        normalization = np.sum(mask, axis=-1, keepdims=True)
        covariance /= normalization
        assert covariance.shape == (*independent, D, D), covariance.shape

        if hermitize:
            covariance = force_hermitian(covariance)

        if trace_norm:
            covariance /= np.einsum(
                '...dd', covariance
            )[..., None, None]

        # if self.use_pinv:
        #     if calculate_covariance:
        #         params.covariance = covariance
        #     params.precision = np.linalg.pinv(covariance)
        #     params.determinant = 1 / np.linalg.det(params.precision)
        #     params.log_determinant = -np.linalg.slogdet(params.precision)[1]
        # else:
        try:
            eigenvals, eigenvecs = np.linalg.eigh(covariance)
        except np.linalg.LinAlgError:
            eigenvals, eigenvecs = np.linalg.eig(covariance)
        eigenvals = eigenvals.real

        eigenvals = eigenvals / np.amax(eigenvals, axis=-1, keepdims=True)

        eigenvals = np.maximum(
            eigenvals,
            eigenvalue_floor
        )

        if calculate_covariance:
            # diagonal = np.einsum('de,...d->...de', np.eye(D), eigenvals)
            params.covariance = np.einsum(
                '...wx,...x,...zx->...wz',
                eigenvecs,
                eigenvals,
                eigenvecs.conj(),
                optimize='greedy',
            )
        # params.determinant = np.prod(eigenvals, axis=-1)
        params.log_determinant = np.sum(np.log(eigenvals), axis=-1)

        params.covariance_eigenvectors = eigenvecs
        params.covariance_eigenvalues = eigenvals

        # inverse_diagonal = np.einsum('de,...d->...de', np.eye(D),
        #                              1 / eigenvals)
        # params.precision = np.einsum(
        #     '...wx,...x,...zx->...wz',
        #     eigenvecs,
        #     1 / eigenvals,
        #     eigenvecs.conj(),
        #     optimize='greedy',
        # )

        return params
