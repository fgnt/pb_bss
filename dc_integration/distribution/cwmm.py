from dataclasses import dataclass
from dataclasses import field

import numpy as np
from .complex_watson import ComplexWatsonParameters, ComplexWatson

from dc_integration.utils import (
    get_power_spectral_density_matrix,
    get_pca,
)
from dc_integration.distribution.util import (
    _unit_norm,
    _Parameter,
)


@dataclass
class ComplexWatsonMixtureModelParameters(_Parameter):
    complex_watson: ComplexWatsonParameters \
        = field(default_factory=ComplexWatsonParameters)
    mixture_weights: np.array = None
    affiliation: np.array = None

    eps: float = 1e-10

    def _predict(self, Y, source_activity_mask=None):
        """Predict class affiliation posteriors from given model.

        Args:
            Y: Normalized mix with shape (..., T, D).
        Returns: Affiliations with shape (..., K, T).
        """
        affiliation = self.mixture_weights[..., None] * ComplexWatson.pdf(
            Y[..., None, :, :],
            np.ascontiguousarray(self.complex_watson.mode[..., None, :]),
            self.complex_watson.concentration[..., None]
        )
        if source_activity_mask is not None:
            affiliation *= source_activity_mask
        affiliation /= np.maximum(
            np.sum(affiliation, axis=-2, keepdims=True),
            self.eps,
        )
        affiliation = np.maximum(affiliation, self.eps)
        return np.ascontiguousarray(affiliation)

    def predict(self, Y):

        *independent, T, D = Y.shape
        assert D < 20, (D, 'Sure?')

        Y = _unit_norm(
            Y,
            axis=-1,
            eps=1e-10,
            eps_style='where'
        )

        return self._predict(Y)


class ComplexWatsonMixtureModel:
    """Collects all functions related to the cWMM."""
    # unit_norm = staticmethod(_unit_norm)
    # phase_norm = staticmethod(_phase_norm)
    # frequency_norm = staticmethod(_frequency_norm)

    Parameters = staticmethod(ComplexWatsonMixtureModelParameters)

    def __init__(self, eps=1e-10, pbar=False):
        """
        """
        self.pbar = pbar
        self.eps = eps

    def fit(
            self,
            Y,
            initialization,
            source_activity_mask=None,
            iterations=100,
            max_concentration=100,
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

        *independent, T, D = Y.shape
        independent = tuple(independent)

        assert D < 20, (D, 'Sure?')

        if isinstance(initialization, self.Parameters):
            K = initialization.mixture_weights.shape[-1]
            assert K < 20, (K, 'Sure?')
        else:
            K = initialization.shape[-2]
            assert K < 20, (K, 'Sure?')
            assert initialization.shape[-1] == T, (initialization.shape, T)
            assert initialization.shape[:-2] == independent, (initialization.shape, independent)

        Y = _unit_norm(
            Y,
            axis=-1,
            eps=1e-10,
            eps_style='where'
        )

        Y_normalized_for_pdf = np.ascontiguousarray(Y)
        Y_normalized_for_psd = np.ascontiguousarray(np.swapaxes(Y, -2, -1))

        if isinstance(initialization, self.Parameters):
            params = initialization
        else:
            params = self.Parameters(eps=self.eps)
            params.affiliation = np.copy(initialization)  # Shape (..., K, T)

        cw = ComplexWatson(D, max_concentration=max_concentration)

        if isinstance(initialization, self.Parameters):
            range_iterations = range(1, 1+iterations)
        else:
            range_iterations = range(iterations)

        if self.pbar:
            import tqdm
            range_iterations = tqdm.tqdm(range_iterations, 'cWMM Iteration')
        else:
            range_iterations = range_iterations

        for i in range_iterations:
            # E step
            if i > 0:
                params.affiliation = params._predict(
                    Y_normalized_for_pdf,
                    source_activity_mask=source_activity_mask,
                )

            # M step
            params.mixture_weights = np.mean(params.affiliation, axis=-1)

            Phi = get_power_spectral_density_matrix(
                Y_normalized_for_psd,
                np.maximum(params.affiliation, params.eps),
                sensor_dim=-2, source_dim=-2, time_dim=-1
            )

            params.complex_watson.mode, eigenvalues = get_pca(Phi)
            params.complex_watson.concentration = \
                cw.hypergeometric_ratio_inverse(eigenvalues)
        return params
