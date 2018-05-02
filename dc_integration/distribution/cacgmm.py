from dataclasses import dataclass, field

import numpy as np
from .complex_angular_central_gaussian import (
    ComplexAngularCentralGaussianParameters,
)


@dataclass
class ComplexAngularCentralGaussianMixtureModelParameters:
    complex_angular_central_gaussian: ComplexAngularCentralGaussianParameters \
        = field(default_factory=ComplexAngularCentralGaussianParameters)
    mixture_weight: np.array = None
    affiliation: np.array = None

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

    def __init__(self, eps=1e-10, visual_debug=False):
        self.eps = eps
        self.visual_debug = visual_debug
        self.covariance = np.empty((), dtype=np.complex)
        self.precision = np.empty((), dtype=np.complex)
        self.determinant = np.empty((), dtype=np.float)
        self.pi = np.empty((), dtype=np.float)

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

