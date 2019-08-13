"""Gaussian complex-Angular-Centric-Gaussian mixture model

This is a specific mixture model to integrate DC and spatial observations. It
does and will not support independent dimensions. This also explains, why
concrete variable names (i.e. F, T, embedding) are used instead of unnamed
independent axes.

The Gaussian distributions are assumed to be spherical (scaled identity).

@article{Drude2019Integration,
  title={Integration of neural networks and probabilistic spatial models for acoustic blind source separation},
  author={Drude, Lukas and Haeb-Umbach, Reinhold},
  journal={IEEE Journal of Selected Topics in Signal Processing},
  year={2019},
  publisher={IEEE}
}
"""
from operator import xor
from typing import Any

import numpy as np
from dataclasses import dataclass
from pb_bss.utils import unsqueeze

from pb_bss.distribution import (
    ComplexAngularCentralGaussian,
    ComplexAngularCentralGaussianTrainer,
)
from pb_bss.distribution import GaussianTrainer
from .mixture_model_utils import (
    log_pdf_to_affiliation,
    log_pdf_to_affiliation_for_integration_models_with_inline_pa,
)
from .utils import _ProbabilisticModel


@dataclass
class GCACGMM(_ProbabilisticModel):
    weight: np.array  # Shape (), (K,), (F, K), (T, K)
    weight_constant_axis: tuple
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

    def _predict(
            self,
            observation,
            embedding,
            affiliation_eps=0.,
            inline_permutation_alignment=False,
    ):
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

        observation_ = observation[..., None, :, :]
        cacg_log_pdf, quadratic_form = self.cacg._log_pdf(
            np.swapaxes(observation_, -1, -2)
        )

        embedding_ = np.reshape(embedding, (1, F * T, E))
        gaussian_log_pdf = self.gaussian.log_pdf(embedding_)
        num_classes = gaussian_log_pdf.shape[0]
        gaussian_log_pdf = np.transpose(
            np.reshape(gaussian_log_pdf, (num_classes, F, T)), (1, 0, 2)
        )

        if inline_permutation_alignment:
            affiliation \
                = log_pdf_to_affiliation_for_integration_models_with_inline_pa(
                    weight=unsqueeze(self.weight, self.weight_constant_axis),
                    spatial_log_pdf=self.spatial_weight * cacg_log_pdf,
                    spectral_log_pdf=self.spectral_weight * gaussian_log_pdf,
                    affiliation_eps=affiliation_eps,
                )
        else:
            affiliation = log_pdf_to_affiliation(
                weight=unsqueeze(self.weight, self.weight_constant_axis),
                log_pdf=(
                    self.spatial_weight * cacg_log_pdf
                    + self.spectral_weight * gaussian_log_pdf
                ),
                affiliation_eps=affiliation_eps,
            )

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
        covariance_norm='eigenvalue',
        eigenvalue_floor=1e-10,
        covariance_type="spherical",
        fixed_covariance=None,
        affiliation_eps=1e-10,
        weight_constant_axis=(-1,),
        spatial_weight=1.,
        spectral_weight=1.,
        inline_permutation_alignment=False,
    ) -> GCACGMM:
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
            fixed_covariance: Learned, if None. If fixed, you need to provide
                a covariance matrix with the correct shape.
            affiliation_eps: Used in M-step to clip saliency.
            weight_constant_axis: Axis, along which weight is constant. The
                axis indices are based on affiliation shape. Consequently:
                (-3, -2, -1) == constant = ''
                (-3, -1) == 'k'
                (-1,) == vanilla == 'fk'
                (-3,) == 'kt'
            spatial_weight:
            spectral_weight:
            inline_permutation_alignment: Bool to enable inline permutation
                alignment for integration models. The idea is to reduce
                disagreement between the spatial and the spectral model.

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
        model = None
        for iteration in range(iterations):
            if model is not None:
                affiliation, quadratic_form = model._predict(
                    observation=observation,
                    embedding=embedding,
                    inline_permutation_alignment=inline_permutation_alignment,
                    affiliation_eps=affiliation_eps,
                )

            model = self._m_step(
                observation,
                embedding,
                quadratic_form,
                affiliation=affiliation,
                saliency=saliency,
                hermitize=hermitize,
                covariance_norm=covariance_norm,
                eigenvalue_floor=eigenvalue_floor,
                covariance_type=covariance_type,
                fixed_covariance=fixed_covariance,
                weight_constant_axis=weight_constant_axis,
                spatial_weight=spatial_weight,
                spectral_weight=spectral_weight
            )

        return model

    def fit_predict(
        self,
        observation,
        embedding,
        initialization=None,
        num_classes=None,
        iterations=100,
        saliency=None,
        hermitize=True,
        covariance_norm='eigenvalue',
        eigenvalue_floor=1e-10,
        covariance_type="spherical",
        fixed_covariance=None,
        affiliation_eps=1e-10,
        weight_constant_axis=(-1,),
        spatial_weight=1.,
        spectral_weight=1.,
        inline_permutation_alignment=False,
    ):
        """Fit a model. Then just return the posterior affiliations."""
        model = self.fit(
            observation=observation,
            embedding=embedding,
            initialization=initialization,
            num_classes=num_classes,
            iterations=iterations,
            saliency=saliency,
            hermitize=hermitize,
            covariance_norm=covariance_norm,
            eigenvalue_floor=eigenvalue_floor,
            covariance_type=covariance_type,
            fixed_covariance=fixed_covariance,
            affiliation_eps=affiliation_eps,
            weight_constant_axis=weight_constant_axis,
            spatial_weight=spatial_weight,
            spectral_weight=spectral_weight,
            inline_permutation_alignment=inline_permutation_alignment,
        )
        return model.predict(observation=observation, embedding=embedding)

    def _m_step(
        self,
        observation,
        embedding,
        quadratic_form,
        affiliation,
        saliency,
        hermitize,
        covariance_norm,
        eigenvalue_floor,
        covariance_type,
        fixed_covariance,
        weight_constant_axis,
        spatial_weight,
        spectral_weight
    ):
        F, T, D = observation.shape
        _, _, E = embedding.shape
        _, K, _ = affiliation.shape

        masked_affiliation = affiliation * saliency[..., None, :]

        if -2 in weight_constant_axis:
            weight = 1 / K
        else:
            weight = np.sum(
                masked_affiliation, axis=weight_constant_axis, keepdims=True
            )
            weight /= np.sum(weight, axis=-2, keepdims=True)
            weight = np.squeeze(weight, axis=weight_constant_axis)

        embedding_ = np.reshape(embedding, (1, F * T, E))
        masked_affiliation_ = np.reshape(
            np.transpose(masked_affiliation, (1, 0, 2)),
            (K, F * T)
        )  # 'fkt->k,ft'
        gaussian = GaussianTrainer()._fit(
            y=embedding_,
            saliency=masked_affiliation_,
            covariance_type=covariance_type,
        )

        if fixed_covariance is not None:
            assert fixed_covariance.shape == gaussian.covariance.shape, (
                f'{fixed_covariance.shape} != {gaussian.covariance.shape}'
            )
            gaussian = gaussian.__class__(
                mean=gaussian.mean,
                covariance=fixed_covariance
            )

        cacg = ComplexAngularCentralGaussianTrainer()._fit(
            y=np.swapaxes(observation[..., None, :, :], -1, -2),
            saliency=masked_affiliation,
            quadratic_form=quadratic_form,
            hermitize=hermitize,
            covariance_norm=covariance_norm,
            eigenvalue_floor=eigenvalue_floor,
        )
        return GCACGMM(
            weight=weight,
            gaussian=gaussian,
            cacg=cacg,
            weight_constant_axis=weight_constant_axis,
            spatial_weight=spatial_weight,
            spectral_weight=spectral_weight
        )
