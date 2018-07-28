from operator import xor

import numpy as np
from dataclasses import dataclass

from dc_integration.distribution import Gaussian, GaussianTrainer


@dataclass
class GMM:
    weight: np.array  # (..., K)
    gaussian: Gaussian

    def predict(self, x):
        *independent, num_observations, _ = x.shape

        affiliation = (
            np.log(self.weight)[..., :, None]
            + self.gaussian.log_pdf(x)
        )
        affiliation -= np.max(affiliation, axis=-2)
        np.exp(affiliation, out=affiliation)
        denominator = np.maximum(
            np.einsum("...kn->...n", affiliation)[..., None, :],
            np.finfo(x.dtype).tiny,
        )
        affiliation /= denominator
        return affiliation


class GMMTrainer:
    def __init__(self, eps=1e-10):
        self.eps = eps
        self.log_likelihood_history = []

    def fit(
        self,
        x,
        initialization=None,
        num_classes=None,
        iterations=100,
        saliency=None,
        covariance_type="full",
    ):
        """

        Args:
            x: Shape (..., N, D)
            initialization: Affiliations between 0 and 1. Shape (..., K, N)
            num_classes: Scalar >0
            iterations: Scalar >0
            saliency: Importance weighting for each observation, shape (..., N)
            covariance_type: Either 'full', 'diagonal', or 'spherical'

        Returns:

        TODO: Support different weight types
        """
        assert xor(initialization is None, num_classes is None), (
            "Incompatible input combination. "
            "Exactly one of the two inputs has to be None: "
            f"{initialization is None} xor {num_classes is None}"
        )

        if initialization is None and num_classes is not None:
            *independent, num_observations, _ = x.shape
            affiliation_shape = (*independent, num_classes, num_observations)
            initialization = np.random.uniform(size=affiliation_shape)
            initialization /= np.einsum("...kn->...n", initialization)[
                ..., None, :
            ]

        if saliency is None:
            saliency = np.ones_like(initialization[..., 0, :])

        return self._fit(
            x,
            initialization=initialization,
            iterations=iterations,
            saliency=saliency,
            covariance_type=covariance_type,
        )

    def _fit(self, x, initialization, iterations, saliency, covariance_type):
        affiliation = initialization  # TODO: Do we need np.copy here?
        for iteration in range(iterations):
            model = self._m_step(
                x,
                affiliation=affiliation,
                saliency=saliency,
                covariance_type=covariance_type,
            )

            if iteration < iterations - 1:
                affiliation = model.predict(x)

        return model

    def _m_step(self, x, affiliation, saliency, covariance_type):
        masked_affiliations = affiliation * saliency[..., None, :]
        weight = np.einsum("...kn->...k", masked_affiliations)
        weight /= np.einsum("...n->...", saliency)[..., None]

        gaussian = GaussianTrainer()._fit(
            x=x, saliency=masked_affiliations, covariance_type=covariance_type
        )
        return GMM(weight=weight, gaussian=gaussian)
