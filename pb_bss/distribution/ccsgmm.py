from operator import xor

import numpy as np
from dataclasses import dataclass

from .complex_circular_symmetric_gaussian import (
    ComplexCircularSymmetricGaussian,
    ComplexCircularSymmetricGaussianTrainer,
)
from .mixture_model_utils import log_pdf_to_affiliation
from .utils import _ProbabilisticModel


@dataclass
class CCSGMM(_ProbabilisticModel):
    weight: np.array  # (..., K)
    complex_gaussian: ComplexCircularSymmetricGaussian

    def predict(self, y):
        """Predict class affiliation posteriors from given model.

        Args:
            y: Observations with shape (..., N, D).
                Observations are expected to are unit norm normalized.
        Returns: Affiliations with shape (..., K, T).
        """
        assert np.iscomplexobj(y), y.dtype
        y = y / np.maximum(
            np.linalg.norm(y, axis=-1, keepdims=True), np.finfo(y.dtype).tiny
        )
        return self._predict(y)

    def _predict(self, y):
        """Predict class affiliation posteriors from given model.

        Args:
            y: Observations with shape (..., N, D).
                Observations are expected to are unit norm normalized.
        Returns: Affiliations with shape (..., K, T).
        """
        return log_pdf_to_affiliation(
                self.weight[..., :, None],
                self.complex_gaussian.log_pdf(y[..., None, :, :]),
                source_activity_mask=None,
                affiliation_eps=0.,
        )


class CCSGMMTrainer:
    def fit(
            self,
            y,
            initialization=None,
            num_classes=None,
            iterations=100,
            saliency=None,
    ) -> CCSGMM:
        """ EM for CGMMs with any number of independent dimensions.

        Does not support sequence lengths.
        Can later be extended to accept more initializations, but for now
        only accepts affiliations (masks) as initialization.

        Args:
            y: Mix with shape (..., T, D).
            initialization: Shape (..., K, T)
            num_classes: Scalar >0
            iterations: Scalar >0
            saliency: Importance weighting for each observation, shape (..., N)
        """
        assert xor(initialization is None, num_classes is None), (
            "Incompatible input combination. "
            "Exactly one of the two inputs has to be None: "
            f"{initialization is None} xor {num_classes is None}"
        )
        assert np.iscomplexobj(y), y.dtype
        assert y.shape[-1] > 1

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
        )

    def _fit(self, y, initialization, iterations, saliency, ) -> CCSGMM:
        affiliation = initialization  # TODO: Do we need np.copy here?
        for iteration in range(iterations):
            if iteration != 0:
                affiliation = model.predict(y)

            model = self._m_step(y, affiliation=affiliation, saliency=saliency)

        return model

    def _m_step(self, y, affiliation, saliency):
        masked_affiliation = affiliation * saliency[..., None, :]
        weight = np.einsum("...kn->...k", masked_affiliation)
        weight /= np.einsum("...n->...", saliency)[..., None]

        complex_gaussian = ComplexCircularSymmetricGaussianTrainer().fit(
            y=y[..., None, :, :],
            saliency=masked_affiliation,
        )
        return CCSGMM(weight=weight, complex_gaussian=complex_gaussian)
