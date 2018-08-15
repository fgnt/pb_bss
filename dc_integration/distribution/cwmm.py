from operator import xor

from dataclasses import dataclass

import numpy as np
from .complex_watson import ComplexWatson, ComplexWatsonTrainer

from dc_integration.distribution.utils import _ProbabilisticModel
from cached_property import cached_property


@dataclass
class CWMM(_ProbabilisticModel):
    weight: np.array  # (..., K)
    complex_watson: ComplexWatson

    def predict(self, x):
        """Predict class affiliation posteriors from given model.

        Args:
            x: Observations with shape (..., N, D).
                Observations are expected to are unit norm normalized.
        Returns: Affiliations with shape (..., K, T).
        """
        assert np.iscomplexobj(x), x.dtype
        x = x / np.maximum(
            np.linalg.norm(x, axis=-1, keepdims=True), np.finfo(x.dtype).tiny
        )
        return self._predict(x)

    def _predict(self, x):
        """Predict class affiliation posteriors from given model.

        Args:
            x: Observations with shape (..., N, D).
                Observations are expected to are unit norm normalized.
        Returns: Affiliations with shape (..., K, T).
        """
        log_pdf = self.complex_watson.pdf(x[..., None, :, :])

        affiliation = np.log(self.weight)[..., :, None] + log_pdf
        affiliation -= np.max(affiliation, axis=-2, keepdims=True)
        np.exp(affiliation, out=affiliation)
        denominator = np.maximum(
            np.einsum("...kn->...n", affiliation)[..., None, :],
            np.finfo(affiliation.dtype).tiny,
        )
        affiliation /= denominator
        return affiliation


class CWMMTrainer:
    def __init__(
        self, dimension=None, max_concentration=100, spline_markers=100
    ):
        """

        Args:
            dimension: Feature dimension. If you do not provide this when
                initializing the trainer, it will be inferred when the fit
                function is called.
            max_concentration: For numerical stability reasons.
            spline_markers:
        """
        self.dimension = dimension
        self.max_concentration = max_concentration
        self.spline_markers = spline_markers

    def fit(
            self,
            x,
            initialization=None,
            num_classes=None,
            iterations=100,
            saliency=None,
    ) -> CWMM:
        """ EM for CWMMs with any number of independent dimensions.

        Does not support sequence lengths.
        Can later be extended to accept more initializations, but for now
        only accepts affiliations (masks) as initialization.

        Args:
            x: Mix with shape (..., T, D).
            initialization: Shape (..., K, T)
            num_classes: Scalar >0
            iterations: Scalar >0
        """
        assert xor(initialization is None, num_classes is None), (
            "Incompatible input combination. "
            "Exactly one of the two inputs has to be None: "
            f"{initialization is None} xor {num_classes is None}"
        )
        assert np.iscomplexobj(x), x.dtype
        x = x / np.maximum(
            np.linalg.norm(x, axis=-1, keepdims=True), np.finfo(x.dtype).tiny
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

        if self.dimension is None:
            self.dimension = x.shape[-1]
        else:
            assert self.dimension == x.shape[-1], (
                "You initialized the trainer with a different dimension than "
                "you are using to fit a model. Use a new trainer, when you "
                "change the dimension."
            )

        return self._fit(
            x,
            initialization=initialization,
            iterations=iterations,
            saliency=saliency,
        )

    def _fit(self, x, initialization, iterations, saliency,) -> CWMM:
        affiliation = initialization  # TODO: Do we need np.copy here?
        for iteration in range(iterations):
            model = self._m_step(x, affiliation=affiliation, saliency=saliency)

            if iteration < iterations - 1:
                affiliation = model.predict(x)

        return model

    @cached_property
    def complex_watson_trainer(self):
        return ComplexWatsonTrainer(
            self.dimension,
            max_concentration=self.max_concentration,
            spline_markers=self.spline_markers
        )

    def _m_step(self, x, affiliation, saliency):
        masked_affiliation = affiliation * saliency[..., None, :]
        weight = np.einsum("...kn->...k", masked_affiliation)
        weight /= np.einsum("...n->...", saliency)[..., None]

        complex_watson = self.complex_watson_trainer._fit(
            x=x,
            saliency=masked_affiliation,
        )
        return CWMM(weight=weight, complex_watson=complex_watson)
