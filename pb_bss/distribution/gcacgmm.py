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

from pb_bss.distribution import GaussianTrainer
from pb_bss.distribution import (
    ComplexAngularCentralGaussian,
    ComplexAngularCentralGaussianTrainer,
)
from pb_bss.distribution.utils import _ProbabilisticModel
from pb_bss.utils import unsqueeze


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

        affiliation = (
            unsqueeze(np.log(self.weight), self.weight_constant_axis)
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
        covariance_norm='eigenvalue',
        eigenvalue_floor=1e-10,
        covariance_type="spherical",
        fixed_covariance=None,
        affiliation_eps=1e-10,
        weight_constant_axis=(-1,),
        spatial_weight=1.,
        spectral_weight=1.
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
                (-1) == vanilla == 'fk'
                (-3) == 'kt'
            spatial_weight:
            spectral_weight:

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
                covariance_norm=covariance_norm,
                eigenvalue_floor=eigenvalue_floor,
                covariance_type=covariance_type,
                fixed_covariance=fixed_covariance,
                weight_constant_axis=weight_constant_axis,
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


class PartiallySharedGCACGMMTrainer(GCACGMMTrainer):
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
        fixed_covariance,
        weight_constant_axis,
        spatial_weight,
        spectral_weight
    ):
        F, T, D = observation.shape
        _, _, E = embedding.shape
        _, K, _ = affiliation.shape

        assert K == 4, f'This fancy sharing is just tested for K == 4 != {K}.'

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

        masked_affiliation_for_gaussian = np.stack((
            masked_affiliation[:, 0, :] + masked_affiliation[:, 1, :],
            masked_affiliation[:, 2, :] + masked_affiliation[:, 3, :]
        ), axis=1)
        masked_affiliation_for_gaussian = np.reshape(
            np.transpose(masked_affiliation_for_gaussian, (1, 0, 2)),
            (2, F * T)
        )  # 'fkt->k,ft'
        gaussian = GaussianTrainer()._fit(
            y=embedding_,
            saliency=masked_affiliation_for_gaussian,
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

        # There are 4 classes in quadratic_form. Entry 1 and 2 are equal.
        assert np.mean(
            np.abs(quadratic_form[:, 1, :] - quadratic_form[:, 2, :])
        ) < 1e-4
        quadratic_form = quadratic_form[:, [0, 1, 3], :]

        masked_affiliation_for_cacg = np.stack((
            masked_affiliation[:, 0, :],
            masked_affiliation[:, 1, :] + masked_affiliation[:, 2, :],
            masked_affiliation[:, 3, :]
        ), axis=1)
        cacg = ComplexAngularCentralGaussianTrainer()._fit(
            y=observation[..., None, :, :],
            saliency=masked_affiliation_for_cacg,
            quadratic_form=quadratic_form,
            hermitize=hermitize,
            trace_norm=trace_norm,
            eigenvalue_floor=eigenvalue_floor,
        )

        # Expand again
        gaussian = gaussian.__class__(
            mean=np.stack((
                gaussian.mean[0, :],
                gaussian.mean[0, :],
                gaussian.mean[1, :],
                gaussian.mean[1, :]
            )),
            covariance=np.stack((
                gaussian.covariance[0],
                gaussian.covariance[0],
                gaussian.covariance[1],
                gaussian.covariance[1],
            ))
        )
        # print('gaussian.mean.shape', gaussian.mean.shape)
        # print('gaussian.covariance.shape', gaussian.covariance.shape)

        # Expand again
        cacg = ComplexAngularCentralGaussian.from_covariance(
            covariance=np.stack((
                cacg.covariance[:, 0, :, :],
                cacg.covariance[:, 1, :, :],
                cacg.covariance[:, 1, :, :],
                cacg.covariance[:, 2, :, :]
            ), axis=1)
        )
        # print('cacg.covariance.shape', cacg.covariance.shape)

        return GCACGMM(
            weight=weight,
            gaussian=gaussian,
            cacg=cacg,
            weight_constant_axis=weight_constant_axis,
            spatial_weight=spatial_weight,
            spectral_weight=spectral_weight
        )
