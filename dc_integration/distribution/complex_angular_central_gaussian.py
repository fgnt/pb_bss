from dataclasses import dataclass

import numpy as np


@dataclass
class ComplexAngularCentralGaussianParameters:
    covariance: np.array = None
    precision: np.array = None
    determinant: np.array = None


class ComplexAngularCentralGaussian:

    def _fit(
            self,
            Y,
            saliency,
            quadratic_form,
            hermitize=True,
            trace_norm=True,
            eigenvalue_floor=1e-10,
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

        """
        assert Y.ndim == saliency.ndim + 1, (Y.shape, saliency.ndim)
        *independent, D, T = Y.shape
        independent = list(independent)

        # Special case for mixture model
        independent[-1] = saliency.shape[-2]
        # K = saliency.shape[-2]

        params = ComplexAngularCentralGaussianParameters()

        mask = saliency[..., None, :]
        assert mask.shape == (*independent, 1, T), (mask.shape, (*independent, 1, T))

        params.covariance = D * np.einsum(
            '...t,...dt,...et->...de',
            (saliency / quadratic_form),
            Y,
            Y.conj()
        )
        normalization = np.sum(mask, axis=-1, keepdims=True)
        params.covariance /= normalization
        assert params.covariance.shape == (*independent, D, D), params.covariance.shape

        if hermitize:
            params.covariance = \
                (
                        params.covariance
                        + np.swapaxes(params.covariance.conj(), -1, -2)
                ) / 2

        if trace_norm:
            params.covariance /= np.einsum(
                '...dd', params.covariance
            )[..., None, None]

        eigenvals, eigenvecs = np.linalg.eigh(params.covariance)
        eigenvals = eigenvals.real
        eigenvals = np.maximum(
            eigenvals,
            np.max(eigenvals, axis=-1, keepdims=True) * eigenvalue_floor
        )
        diagonal = np.einsum('de,...d->...de', np.eye(D), eigenvals)
        params.covariance = np.einsum(
            '...wx,...xy,...zy->...wz', eigenvecs, diagonal,
            eigenvecs.conj()
        )
        params.determinant = np.prod(eigenvals, axis=-1)
        inverse_diagonal = np.einsum('de,...d->...de', np.eye(D),
                                     1 / eigenvals)
        params.precision = np.einsum(
            '...wx,...xy,...zy->...wz', eigenvecs, inverse_diagonal,
            eigenvecs.conj()
        )

        return params
