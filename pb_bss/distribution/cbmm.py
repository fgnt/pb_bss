from operator import xor
from cached_property import cached_property
from dataclasses import dataclass

import numpy as np
from .complex_bingham import (
    ComplexBingham,
    ComplexBinghamTrainer,
    normalize_observation,
)

from pb_bss.distribution.utils import (
    _ProbabilisticModel,
    estimate_mixture_weight,
)
from pb_bss.distribution.mixture_model_utils import log_pdf_to_affiliation


@dataclass
class CBMM(_ProbabilisticModel):
    weight: np.array  # (..., K)
    complex_bingham: ComplexBingham

    def predict(self, y, affiliation_eps=0):
        """Predict class affiliation posteriors from given model.

        Args:
            y: Observations with shape (..., N, D).
                Observations are expected to are unit norm normalized.
            affiliation_eps:
        Returns: Affiliations with shape (..., K, T).
        """
        assert np.iscomplexobj(y), y.dtype
        y = y / np.maximum(
            np.linalg.norm(y, axis=-1, keepdims=True), np.finfo(y.dtype).tiny
        )
        return self._predict(y, affiliation_eps=affiliation_eps)

    def _predict(self, y, affiliation_eps):
        """Predict class affiliation posteriors from given model.

        Args:
            y: Observations with shape (..., N, D).
                Observations are expected to are unit norm normalized.
            affiliation_eps:
        Returns: Affiliations with shape (..., K, T).
        """
        affiliation = log_pdf_to_affiliation(
                self.weight,
                self.complex_bingham.log_pdf(y[..., None, :, :]),
                source_activity_mask=None,
                affiliation_eps=affiliation_eps,
        )

        return affiliation


class CBMMTrainer:
    def __init__(
            self,
            dimension=None,
            max_concentration=np.inf,
            eigenvalue_eps=1e-8,
    ):
        """

        Should we introduce something like max_concentration as in watson?

        Args:
            dimension: Feature dimension. If you do not provide this when
                initializing the trainer, it will be inferred when the fit
                function is called.

        """
        self.dimension = dimension
        self.max_concentration = max_concentration
        self.eigenvalue_eps = eigenvalue_eps

    def fit(
            self,
            y,
            initialization=None,
            num_classes=None,
            iterations=100,
            *,
            saliency=None,
            weight_constant_axis=(-1,),
            affiliation_eps=0,
            return_affiliation=False,
    ) -> CBMM:
        """ EM for CBMMs with any number of independent dimensions.

        Does not support sequence lengths.
        Can later be extended to accept more initializations, but for now
        only accepts affiliations (masks) as initialization.

        Args:
            y: Mix with shape (..., T, D).
            initialization: Shape (..., K, T)
            num_classes: Scalar >0
            iterations: Scalar >0
            saliency: Importance weighting for each observation, shape (..., N)
            weight_constant_axis: 
            affiliation_eps: 
            return_affiliation:
        """
        assert xor(initialization is None, num_classes is None), (
            "Incompatible input combination. "
            "Exactly one of the two inputs has to be None: "
            f"{initialization is None} xor {num_classes is None}"
        )
        assert np.iscomplexobj(y), y.dtype
        assert y.shape[-1] > 1

        y = normalize_observation(y)

        if initialization is None and num_classes is not None:
            *independent, num_observations, _ = y.shape
            affiliation_shape = (*independent, num_classes, num_observations)
            initialization = np.random.uniform(size=affiliation_shape)
            initialization /= np.einsum("...kn->...n", initialization)[
                ..., None, :
            ]

        if saliency is None:
            saliency = np.ones_like(initialization[..., 0, :])

        if self.dimension is None:
            self.dimension = y.shape[-1]
        else:
            assert self.dimension == y.shape[-1], (
                "You initialized the trainer with a different dimension than "
                "you are using to fit a model. Use a new trainer, when you "
                "change the dimension."
            )

        return self._fit(
            y,
            initialization=initialization,
            iterations=iterations,
            saliency=saliency,
            affiliation_eps=affiliation_eps,
            return_affiliation=return_affiliation,
            weight_constant_axis=weight_constant_axis,
        )

    def _fit(
            self,
            y,
            initialization,
            iterations,
            saliency,
            return_affiliation,
            weight_constant_axis,
            affiliation_eps,
    ) -> CBMM:
        # assert affiliation_eps == 0, affiliation_eps
        affiliation = initialization  # TODO: Do we need np.copy here?
        for iteration in range(iterations):
            if iteration != 0:
                affiliation = model.predict(y, affiliation_eps=affiliation_eps)

            model = self._m_step(
                y,
                affiliation=affiliation,
                saliency=saliency,
                weight_constant_axis=weight_constant_axis,
            )

        if return_affiliation is True:
            return model, affiliation
        elif return_affiliation is False:
            return model
        else:
            raise ValueError(return_affiliation)

    @cached_property
    def complex_bingham_trainer(self):
        return ComplexBinghamTrainer(
            self.dimension,
            max_concentration=self.max_concentration,
            eignevalue_eps=self.eigenvalue_eps,
        )

    def _m_step(
            self,
            y,
            affiliation,
            saliency,
            weight_constant_axis,
    ):
        weight = estimate_mixture_weight(
            affiliation=affiliation,
            saliency=saliency,
            weight_constant_axis=weight_constant_axis,
        )

        if saliency is None:
            masked_affiliation = affiliation
        else:
            masked_affiliation = affiliation * saliency[..., None, :]

        complex_bingham = self.complex_bingham_trainer._fit(
            y=y[..., None, :, :],
            saliency=masked_affiliation,
        )
        return CBMM(weight=weight, complex_bingham=complex_bingham)
