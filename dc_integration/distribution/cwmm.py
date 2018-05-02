from dataclasses import dataclass
from dataclasses import field

import numpy as np
from .complex_watson import ComplexWatsonParameters

@dataclass
class ComplexWatsonMixtureModelParameters:
    complex_watson: ComplexWatsonParameters = field(default_factory=ComplexWatsonParameters)
    mixture_weights: np.array = None
    affiliation: np.array = None

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


class ComplexWatsonMixtureModel:
    """Collects all functions related to the cWMM."""
    unit_norm = staticmethod(_unit_norm)
    phase_norm = staticmethod(_phase_norm)
    frequency_norm = staticmethod(_frequency_norm)

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

    def fit(
            self, Y_normalized, initialization,
            iterations=100, max_concentration=100
    ) -> ComplexWatsonMixtureModelParameters:
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
