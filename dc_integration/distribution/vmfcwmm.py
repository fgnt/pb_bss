from dataclasses import dataclass

import numpy as np
from .complex_watson import ComplexWatsonParameters
from .von_mises_fisher import VonMisesFisherParameters


@dataclass
class VonMisesFisherComplexWatsonMixtureModelParameters:
    von_mises_fisher: VonMisesFisherParameters = field(default_factory=VonMisesFisherParameters)
    complex_watson: ComplexWatsonParameters = field(default_factory=ComplexWatsonParameters)
    von_mises_fisher_score: float = None
    complex_gaussian_score: float = None
    mixture_weights: np.array = None
    affiliation: np.array = None

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


class VonMisesFisherComplexWatsonMixtureModel:
    """Hybrid model."""
    unit_norm = staticmethod(_unit_norm)

    def __init__(self, *, spatial_score, embedding_score):
        self.spatial_score = spatial_score
        self.embedding_score = embedding_score
        self.mu = np.empty((), dtype=np.float)
        self.kappa_vmf = np.empty((), dtype=np.float)
        self.pi = np.empty((), dtype=np.float)
        self.W = np.empty((), dtype=np.float)
        self.kappa_cw = np.empty((), dtype=np.float)

    def fit(
            self, Y, embedding, initialization, iterations=100,
            max_concentration_cw=100, max_concentration_vmf=500
    ) -> VonMisesFisherComplexWatsonMixtureModelParameters:
        """

        Args:
            Y: Mix with shape (..., T, D).
            embedding: Embedding from Deep Clustering with shape (..., T, E).
            initialization: Shape (..., K, T)
            iterations:
            max_concentration_cw:
            max_concentration_vmf:

        Returns:

        """
        # TODO: Normalize Y

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
