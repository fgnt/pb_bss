from dataclasses import dataclass
from dataclasses import field

import numpy as np


@dataclass
class VonMisesFisherComplexAngularCentralGaussianMixtureModelParameters:
    von_mises_fisher: VonMisesFisherParameters = field(default_factory=VonMisesFisherParameters)
    complex_angular_central_gaussian: ComplexAngularCentralGaussianParameters = field(default_factory=ComplexAngularCentralGaussianParameters)
    von_mises_fisher_score: float = None
    complex_angular_central_gaussian_score: float = None
    mixture_weights: np.array = None
    affiliation: np.array = None


class VonMisesFisherComplexAngularCentralGaussianMixtureModel:
    """Hybrid model."""
    unit_norm = staticmethod(_unit_norm)

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

    def fit(
            self, Y, embedding, initialization, iterations=100,
            min_concentration_vmf=0, max_concentration_vmf=500,
            eigenvalue_floor=1e-10
    ):
        """Fit a vMFcACGMM.

        Args:
            Y: Mix with shape (F, T, D).
            embedding: Embedding from Deep Clustering with shape (F*T, E).
            initialization: Shape (F, K, T)
            iterations: Most of the time 10 iterations are acceptable.
            min_concentration_vmf: For numerical stability reasons.
            max_concentration_vmf: For numerical stability reasons.

        Returns:
        """
        F, T, D = Y.shape
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

    def predict(self, Y, embedding):
        """Predict class affiliation posteriors from given model.

        Args:
            Y: Mix with shape (..., T, D).
            embedding: Embedding from Deep Clustering with shape (F*T, E).
        Returns: Affiliations with shape (..., K, T).
        """
        return self._predict(Y, embedding)[0]
