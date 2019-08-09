from operator import xor

import numpy as np
from dataclasses import dataclass
from pb_bss.distribution.utils import _ProbabilisticModel

from .mixture_model_utils import log_pdf_to_affiliation
from .von_mises_fisher import VonMisesFisher, VonMisesFisherTrainer


@dataclass
class VMFMM(_ProbabilisticModel):
    vmf: VonMisesFisher
    weight: np.array  # (..., K)

    def predict(self, y):
        """Predict class affiliation posteriors from given model.

        Args:
            y: Observations with shape (..., N, D).
                Observations are expected to are unit norm normalized.
        Returns: Affiliations with shape (..., K, T).
        """
        assert np.isrealobj(y), y.dtype
        y = y / np.maximum(
            np.linalg.norm(y, axis=-1, keepdims=True), np.finfo(y.dtype).tiny
        )
        return self._predict(y)

    def _predict(self, y):
        return log_pdf_to_affiliation(
                self.weight[..., :, None],
                self.vmf.log_pdf(y[..., None, :, :]),
        )


class VMFMMTrainer:
    """The vMFMM can be used to cluster the embeddings."""

    def fit(
        self,
        y,
        initialization=None,
        num_classes=None,
        iterations=100,
        saliency=None,
        min_concentration=1e-10,
        max_concentration=500,
    ) -> VMFMM:
        """ EM for vMFMMs with any number of independent dimensions.

        Args:
            y: Observations with shape (N, D).
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
        assert np.isrealobj(y), y.dtype
        y = y / np.maximum(
            np.linalg.norm(y, axis=-1, keepdims=True), np.finfo(y.dtype).tiny
        )

        if initialization is None and num_classes is not None:
            *independent, num_observations, _ = y.shape
            affiliation_shape = (*independent, num_classes, num_observations)
            initialization = np.random.uniform(size=affiliation_shape)
            initialization /= np.einsum("...kn->...n", initialization)[
                ..., None, :
            ]

        if saliency is None:
            saliency = np.ones_like(initialization[..., 0, :])

        return self._fit(
            y,
            initialization=initialization,
            iterations=iterations,
            saliency=saliency,
            min_concentration=min_concentration,
            max_concentration=max_concentration,
        )

    def _fit(
        self,
        y,
        initialization,
        iterations,
        saliency,
        min_concentration,
        max_concentration,
    ):
        affiliation = initialization  # TODO: Do we need np.copy here?
        for iteration in range(iterations):
            model = self._m_step(
                y,
                affiliation=affiliation,
                saliency=saliency,
                min_concentration=min_concentration,
                max_concentration=max_concentration,
            )

            if iteration < iterations - 1:
                affiliation = model.predict(y)

        return model

    def _m_step(
        self, y, affiliation, saliency, min_concentration, max_concentration
    ):
        masked_affiliation = affiliation * saliency[..., None, :]
        weight = np.einsum("...kn->...k", masked_affiliation)
        weight /= np.einsum("...n->...", saliency)[..., None]

        vmf = VonMisesFisherTrainer()._fit(
            y=y[..., None, :, :],
            saliency=masked_affiliation,
            min_concentration=min_concentration,
            max_concentration=max_concentration,
        )
        return VMFMM(weight=weight, vmf=vmf)
