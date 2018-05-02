from dataclasses import dataclass, field

import numpy as np
from .complex_angular_central_gaussian import (
    ComplexAngularCentralGaussianParameters,
)
from .util import (
    _unit_norm,
)


@dataclass
class ComplexAngularCentralGaussianMixtureModelParameters:
    cacg: ComplexAngularCentralGaussianParameters \
        = field(default_factory=ComplexAngularCentralGaussianParameters)
    mixture_weight: np.array = None
    affiliation: np.array = None

    eps: float = 1e-10

    def _predict(self, Y):
        """Predict class affiliation posteriors from given model.

        Args:
            Y: Normalized observations with shape (..., T, D).
        Returns: Affiliations with shape (..., K, T) and quadratic format
            with the same shape.
        """
        F, T, D = Y.shape
        K = self.mixture_weight.shape[-1]
        D = Y.shape[-1]

        quadratic_form = np.abs(
            np.einsum(
                '...td,...kde,...te->...kt',
                Y.conj(),
                self.cacg.precision,
                Y,
            )
        ) + self.eps
        assert quadratic_form.shape == (F, K, T), quadratic_form.shape

        affiliation = - D * np.log(quadratic_form)
        affiliation -= np.log(self.cacg.determinant)[..., None]
        affiliation = np.exp(affiliation)
        affiliation *= self.mixture_weight[..., None]

        # ToDo: Figure out if
        # >>> affiliations /= np.sum(affiliations, axis=-2, keepdims=True) + self.eps
        # >>> affiliations = np.clip(affiliations, self.eps, 1 - self.eps)
        # or
        # >>> affiliations = np.clip(affiliations, self.eps, 1 - self.eps)
        # >>> affiliations /= np.sum(affiliations, axis=-2, keepdims=True) + self.eps
        # is better
        affiliation /= np.sum(affiliation, axis=-2, keepdims=True) + self.eps
        affiliation = np.clip(affiliation, self.eps, 1 - self.eps)

        # if self.visual_debug:
        #     _plot_affiliations(affiliations)
        return affiliation, quadratic_form

    def predict(self, Y):
        """Predict class affiliation posteriors from given model.

        Args:
            Y: Normalized observations with shape (..., T, D).
        Returns: Affiliations with shape (..., K, T).
        """
        return self._predict(Y)[0]


class ComplexAngularCentralGaussianMixtureModel:
    """Ito 2016."""
    unit_norm = staticmethod(_unit_norm)

    def __init__(self, eps=1e-10, visual_debug=False, pbar=False):
        self.eps = eps
        self.visual_debug = visual_debug
        # self.covariance = np.empty((), dtype=np.complex)
        # self.precision = np.empty((), dtype=np.complex)
        # self.determinant = np.empty((), dtype=np.float)
        # self.pi = np.empty((), dtype=np.float)
        self.pbar = pbar

    def fit(
            self, Y, initialization, iterations=100,
            hermitize=True, trace_norm=True, eigenvalue_floor=1e-10
    ) -> ComplexAngularCentralGaussianMixtureModelParameters:
        """Fit a cACGMM.

        Args:
            Y: Normalized observations with shape (..., T, D).
            iterations:
            initialization: Shape (..., K, T).
        """

        *independent, T, D = Y.shape
        independent = tuple(independent)
        K = initialization.shape[-2]
        assert K < 20, (K, 'Sure?')
        assert D < 20, (D, 'Sure?')
        assert initialization.shape[-1] == T, (initialization.shape, T)
        assert initialization.shape[:-2] == independent, (initialization.shape, independent)

        Y = _unit_norm(
            Y,
            axis=-1,
            eps=1e-10,
            eps_style='where'
        )

        Y_for_psd = np.ascontiguousarray(np.swapaxes(Y, -2, -1))
        # Y_for_psd: Shape (..., T, K)
        Y_for_pdf = np.ascontiguousarray(Y)

        params = ComplexAngularCentralGaussianMixtureModelParameters(
            eps=self.eps
        )

        params.affiliation = np.copy(initialization)  # Shape (..., K, T)
        quadratic_form = np.ones_like(params.affiliation)  # Shape (..., K, T)

        if self.pbar:
            import tqdm
            range_iterations = tqdm.tqdm(range(iterations), 'cACGMM Iteration')
        else:
            range_iterations = range(iterations)

        for i in range_iterations:
            # E step
            if i > 0:
                # Equation 12
                params.affiliation, quadratic_form = params._predict(Y_for_pdf)

            params.mixture_weight = np.mean(params.affiliation, axis=-1)
            assert params.mixture_weight.shape == (*independent, K), params.mixture_weight.shape

            mask = params.affiliation[..., None, :]
            assert mask.shape == (*independent, K, 1, T), mask.shape
            params.cacg.covariance = D * np.einsum(
                '...kt,...dt,...et->...kde',
                (params.affiliation / quadratic_form),
                Y_for_psd,
                Y_for_psd.conj()
            )
            normalization = np.sum(mask, axis=-1, keepdims=True)
            params.cacg.covariance /= normalization
            assert params.cacg.covariance.shape == (*independent, K, D, D), params.cacg.covariance.shape

            if hermitize:
                params.cacg.covariance = \
                    (
                            params.cacg.covariance
                            + np.swapaxes(params.cacg.covariance.conj(), -1, -2)
                    ) / 2

            if trace_norm:
                params.cacg.covariance /= np.einsum(
                    '...dd', params.cacg.covariance
                )[..., None, None]

            eigenvals, eigenvecs = np.linalg.eigh(params.cacg.covariance)
            eigenvals = eigenvals.real
            eigenvals = np.maximum(
                eigenvals,
                np.max(eigenvals, axis=-1, keepdims=True) * eigenvalue_floor
            )
            diagonal = np.einsum('de,...kd->...kde', np.eye(D), eigenvals)
            params.cacg.covariance = np.einsum(
                '...kwx,...kxy,...kzy->...kwz', eigenvecs, diagonal, eigenvecs.conj()
            )
            params.cacg.determinant = np.prod(eigenvals, axis=-1)
            inverse_diagonal = np.einsum('de,...kd->...kde', np.eye(D),
                                         1 / eigenvals)
            params.cacg.precision = np.einsum(
                '...kwx,...kxy,...kzy->...kwz', eigenvecs, inverse_diagonal,
                eigenvecs.conj()
            )
        return params

            # if self.visual_debug:
            #     with context_manager(figure_size=(24, 3)):
            #         plt.plot(np.log10(
            #             np.max(eigenvals, axis=-1)
            #             / np.min(eigenvals, axis=-1)
            #         ))
            #         plt.xlabel('frequency bin')
            #         plt.ylabel('eigenvalue spread')
            #         plt.show()
            #     with context_manager(figure_size=(24, 3)):
            #         plt.plot(self.pi)
            #         plt.show()

