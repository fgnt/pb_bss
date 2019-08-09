from operator import xor

import numpy as np
from cached_property import cached_property
from dataclasses import dataclass

from .complex_watson import (
    ComplexWatson,
    ComplexWatsonTrainer,
    normalize_observation,
)
from .mixture_model_utils import log_pdf_to_affiliation
from .utils import _ProbabilisticModel, estimate_mixture_weight


@dataclass
class CWMM(_ProbabilisticModel):
    weight: np.array  # (..., K)
    complex_watson: ComplexWatson

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
                self.weight,
                self.complex_watson.log_pdf(y[..., None, :, :]),
                source_activity_mask=None,
                affiliation_eps=0.,
        )


class CWMMTrainer:
    def __init__(
        self, dimension=None, max_concentration=500, spline_markers=1000
    ):
        """

        Args:
            dimension: Feature dimension. If you do not provide this when
                initializing the trainer, it will be inferred when the fit
                function is called.
            max_concentration: For numerical stability reasons.
                500 is relative stable (works for dimension <= 60)
                700 works for dimension <= 7
                800 does not work in the moment
            spline_markers:
        """
        self.dimension = dimension
        self.max_concentration = max_concentration
        self.spline_markers = spline_markers

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
    ) -> CWMM:
        """ EM for CWMMs with any number of independent dimensions.

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
    ) -> CWMM:
        assert affiliation_eps == 0, affiliation_eps
        affiliation = initialization  # TODO: Do we need np.copy here?
        for iteration in range(iterations):
            if iteration != 0:
                affiliation = model.predict(y)

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
    def complex_watson_trainer(self):
        return ComplexWatsonTrainer(
            self.dimension,
            max_concentration=self.max_concentration,
            spline_markers=self.spline_markers
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
            # dirichlet_prior_concentration=dirichlet_prior_concentration,
        )

        if saliency is None:
            masked_affiliation = affiliation
        else:
            masked_affiliation = affiliation * saliency[..., None, :]

        complex_watson = self.complex_watson_trainer._fit(
            y=y[..., None, :, :],
            saliency=masked_affiliation,
        )
        return CWMM(weight=weight, complex_watson=complex_watson)
