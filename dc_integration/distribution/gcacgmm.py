"""Gaussian complex-Angular-Centric-Gaussian mixture model

This is a specific mixture model to integrate DANs and spatial observations. It
does and will not support independent dimensions.

This also explains, why concrete variable names (i.e. F, T, embedding) are used.

Maybe, it is not so nice, that the Gaussians are assumed to be spherical by
default.
"""

from operator import xor
from typing import Any

import numpy as np
from dataclasses import dataclass

from dc_integration.distribution import GaussianTrainer
from dc_integration.distribution import (
    ComplexAngularCentralGaussian,
    ComplexAngularCentralGaussianTrainer,
)
from dc_integration.distribution.utils import _ProbabilisticModel


@dataclass
class GCACGMM(_ProbabilisticModel):
    weight: np.array  # (K,)
    gaussian: Any  # Gaussian, DiagonalGaussian, or SphericalGaussian
    cacg: ComplexAngularCentralGaussian
    spatial_weight: float
    spectral_weight: float

    def predict(self, observation, embedding):
        """

        Args:
            observation: Shape (F, T, D)
            embedding: Shape (F, T, E)

        Returns:
            affiliation: Shape (F, K, T)

        """
        assert np.iscomplexobj(observation), observation.dtype
        assert np.isrealobj(embedding), embedding.dtype
        observation = observation / np.maximum(
            np.linalg.norm(observation, axis=-1, keepdims=True),
            np.finfo(observation.dtype).tiny,
        )
        affiliation, quadratic_form = self._predict(observation, embedding)
        return affiliation

    def _predict(self, observation, embedding):
        """

        Args:
            observation: Shape (F, T, D)
            embedding: Shape (F, T, E)

        Returns:
            affiliation: Shape (F, K, T)
            quadratic_form: Shape (F, K, T)

        """
        F, T, D = observation.shape
        _, _, E = embedding.shape
        num_classes = self.weight.shape[-1]

        observation_ = observation[..., None, :, :]
        cacg_log_pdf, quadratic_form = self.cacg._log_pdf(observation_)

        embedding_ = np.reshape(embedding, (1, F * T, E))
        gaussian_log_pdf = self.gaussian.log_pdf(embedding_)
        gaussian_log_pdf = np.transpose(
            np.reshape(gaussian_log_pdf, (num_classes, F, T)), (1, 0, 2)
        )

        affiliation = (
            np.log(self.weight)[..., :, None]
            + self.spatial_weight * cacg_log_pdf
            + self.spectral_weight * gaussian_log_pdf
        )
        affiliation -= np.max(affiliation, axis=-2, keepdims=True)
        np.exp(affiliation, out=affiliation)
        denominator = np.maximum(
            np.einsum("...kt->...t", affiliation)[..., None, :],
            np.finfo(affiliation.dtype).tiny,
        )
        affiliation /= denominator
        return affiliation, quadratic_form


class GCACGMMTrainer:
    def fit(
        self,
        observation,
        embedding,
        initialization=None,
        num_classes=None,
        iterations=100,
        saliency=None,
        hermitize=True,
        trace_norm=True,
        eigenvalue_floor=1e-10,
        covariance_type="spherical",
        affiliation_eps=1e-10,
        spatial_weight=1.,
        spectral_weight=1.
    ):
        """

        Args:
            observation: Shape (F, T, D)
            embedding: Shape (F, T, E)
            initialization: Affiliations between 0 and 1. Shape (F, K, T)
            num_classes: Scalar >0
            iterations: Scalar >0
            saliency: Importance weighting for each observation, shape (F, T)
            hermitize:
            trace_norm:
            eigenvalue_floor:
            covariance_type: Either 'full', 'diagonal', or 'spherical'
            affiliation_eps: Used in M-step to clip saliency.

        Returns:

        """
        assert xor(initialization is None, num_classes is None), (
            "Incompatible input combination. "
            "Exactly one of the two inputs has to be None: "
            f"{initialization is None} xor {num_classes is None}"
        )
        assert np.iscomplexobj(observation), observation.dtype
        assert np.isrealobj(embedding), embedding.dtype
        assert observation.shape[-1] > 1
        observation = observation / np.maximum(
            np.linalg.norm(observation, axis=-1, keepdims=True),
            np.finfo(observation.dtype).tiny,
        )

        F, T, D = observation.shape
        _, _, E = embedding.shape

        if initialization is None and num_classes is not None:
            affiliation_shape = (F, num_classes, T)
            initialization = np.random.uniform(size=affiliation_shape)
            initialization /= np.einsum("...kt->...t", initialization)[
                ..., None, :
            ]

        if saliency is None:
            saliency = np.ones_like(initialization[..., 0, :])

        quadratic_form = np.ones_like(initialization)
        affiliation = initialization
        for iteration in range(iterations):
            model = self._m_step(
                observation,
                embedding,
                quadratic_form,
                affiliation=np.clip(
                    affiliation, affiliation_eps, 1 - affiliation_eps
                ),
                saliency=saliency,
                hermitize=hermitize,
                trace_norm=trace_norm,
                eigenvalue_floor=eigenvalue_floor,
                covariance_type=covariance_type,
                spatial_weight=spatial_weight,
                spectral_weight=spectral_weight
            )

            if iteration < iterations - 1:
                affiliation, quadratic_form = model._predict(
                    observation=observation, embedding=embedding
                )

        return model

    def _m_step(
        self,
        observation,
        embedding,
        quadratic_form,
        affiliation,
        saliency,
        hermitize,
        trace_norm,
        eigenvalue_floor,
        covariance_type,
        spatial_weight,
        spectral_weight
    ):
        F, T, D = observation.shape
        _, _, E = embedding.shape
        _, K, _ = affiliation.shape

        masked_affiliation = affiliation * saliency[..., None, :]
        weight = np.einsum("fkt->fk", masked_affiliation)
        weight /= np.einsum("ft->f", saliency)[..., None]

        embedding_ = np.reshape(embedding, (1, F * T, E))
        masked_affiliation_ = np.reshape(
            np.transpose(masked_affiliation, (1, 0, 2)),
            (K, F * T)
        )  # 'fkt->k,ft'
        gaussian = GaussianTrainer()._fit(
            x=embedding_,
            saliency=masked_affiliation_,
            covariance_type=covariance_type,
        )
        cacg = ComplexAngularCentralGaussianTrainer()._fit(
            x=observation[..., None, :, :],
            saliency=masked_affiliation,
            quadratic_form=quadratic_form,
            hermitize=hermitize,
            trace_norm=trace_norm,
            eigenvalue_floor=eigenvalue_floor,
        )
        return GCACGMM(
            weight=weight,
            gaussian=gaussian,
            cacg=cacg,
            spatial_weight=spatial_weight,
            spectral_weight=spectral_weight
        )
