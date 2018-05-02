from dataclasses import dataclass
from dataclasses import field

import numpy as np


@dataclass
class VonMisesFisherComplexGaussianMixtureModelParameters:
    von_mises_fisher: VonMisesFisherParameters = field(default_factory=VonMisesFisherParameters)
    complex_gaussian: ComplexGaussianParameters = field(default_factory=ComplexGaussianParameters)
    von_mises_fisher_score: float = None
    complex_gaussian_score: float = None
    mixture_weights: np.array = None
    affiliation: np.array = None


class VonMisesFisherComplexGaussianMixtureModel:
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
        self.pi = np.empty((), dtype=np.float)
        self.covariance = np.empty((), dtype=np.float)
        self.precision = np.empty((), dtype=np.float)
        self.determinant = np.empty((), dtype=np.float)

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

    def predict(self, Y, embedding, inverse='inv'):
        """Predict class affiliation posteriors from given model.

        Args:
            Y: Mix with shape (..., T, D).
            embedding: Embedding from Deep Clustering with shape (F*T, E).
        Returns: Affiliations with shape (..., K, T).
        """
        return self._predict(Y, embedding, inverse=inverse)[0]
