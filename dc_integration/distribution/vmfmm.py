from operator import xor

import numpy as np
from dataclasses import dataclass

from .von_mises_fisher import VonMisesFisher, VonMisesFisherTrainer


@dataclass
class VMFMM:
    vmf: VonMisesFisher
    weight: np.array  # (..., K)

    def predict(self, x):
        """Predict class affiliation posteriors from given model.

        Args:
            x: Observations with shape (..., N, D).
                Observations are expected to are unit norm normalized.
        Returns: Affiliations with shape (..., K, T).
        """
        assert np.isrealobj(x), x.dtype
        x = x / np.maximum(
            np.linalg.norm(x, axis=-1, keepdims=True), np.finfo(x.dtype).tiny
        )
        return self._predict(x)

    def _predict(self, x):
        log_pdf = self.vmf.pdf(x[..., None, :, :])

        affiliation = np.log(self.weight)[..., :, None] + log_pdf
        affiliation -= np.max(affiliation, axis=-2, keepdims=True)
        np.exp(affiliation, out=affiliation)
        denominator = np.maximum(
            np.einsum("...kn->...n", affiliation)[..., None, :],
            np.finfo(affiliation.dtype).tiny,
        )
        affiliation /= denominator
        return affiliation


class VMFMMTrainer:
    """The vMFMM can be used to cluster the embeddings."""

    def fit(
        self,
        x,
        initialization=None,
        num_classes=None,
        iterations=100,
        saliency=None,
        min_concentration=1e-10,
        max_concentration=500,
    ) -> VMFMM:
        """ EM for vMFMMs with any number of independent dimensions.

        Args:
            x: Observations with shape (N, D).
            initialization: Affiliations between 0 and 1. Shape (..., K, N)
            num_classes: Scalar >0
            iterations: Scalar >0
            saliency: Importance weighting for each observation, shape (..., N)
            min_concentration:
            max_concentration:
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
            min_concentration=min_concentration,
            max_concentration=max_concentration,
        )

    def _fit(
        self,
        x,
        initialization,
        iterations,
        saliency,
        min_concentration,
        max_concentration,
    ):
        affiliation = initialization  # TODO: Do we need np.copy here?
        for iteration in range(iterations):
            model = self._m_step(
                x,
                affiliation=affiliation,
                saliency=saliency,
                min_concentration=min_concentration,
                max_concentration=max_concentration,
            )

            if iteration < iterations - 1:
                affiliation = model.predict(x)

        return model

    def _m_step(
        self, x, affiliation, saliency, min_concentration, max_concentration
    ):
        masked_affiliation = affiliation * saliency[..., None, :]
        weight = np.einsum("...kn->...k", masked_affiliation)
        weight /= np.einsum("...n->...", saliency)[..., None]

        vmf = VonMisesFisherTrainer()._fit(
            x=x,
            saliency=masked_affiliation,
            min_concentration=min_concentration,
            max_concentration=max_concentration,
        )
        return VMFMM(weight=weight, vmf=vmf)
