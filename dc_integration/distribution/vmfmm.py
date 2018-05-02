from dataclasses import dataclass
from dataclasses import field

import numpy as np
from .von_mises_fisher import VonMisesFisherParameters


@dataclass
class VonMisesFisherMixtureModelParameters:
    von_mises_fisher: VonMisesFisherParameters = field(default_factory=VonMisesFisherParameters)
    mixture_weights: np.array = None
    affiliation: np.array = None


class VonMisesFisherMixtureModel:
    """The vMFMM can be used to cluster the embeddings."""
    unit_norm = staticmethod(_unit_norm)

    def __init__(self, eps=1e-10, visual_debug=False):
        self.eps = eps
        self.visual_debug = visual_debug
        self.mu = None
        self.kappa = None
        self.pi = None

    def fit(
            self, x, initialization,
            iterations=100, min_concentration=0, max_concentration=500
    ) -> VonMisesFisherMixtureModelParameters:
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
