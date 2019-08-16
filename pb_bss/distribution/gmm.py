from operator import xor

import numpy as np
from dataclasses import dataclass
from pb_bss.distribution.mixture_model_utils import (
    estimate_mixture_weight,
    log_pdf_to_affiliation,
)
from pb_bss.utils import labels_to_one_hot
from sklearn.cluster import KMeans

from . import Gaussian, GaussianTrainer
from .utils import _ProbabilisticModel


@dataclass
class GMM(_ProbabilisticModel):
    weight: np.array  # (..., K)
    gaussian: Gaussian

    def predict(self, x):
        return log_pdf_to_affiliation(
            self.weight,
            self.gaussian.log_pdf(x[..., None, :, :]),
        )


class GMMTrainer:
    def __init__(self, eps=1e-10):
        self.eps = eps
        self.log_likelihood_history = []

    def fit(
        self,
        y,
        initialization=None,
        num_classes=None,
        iterations=100,
        *,
        saliency=None,
        weight_constant_axis=(-1,),
        covariance_type="full",
        fixed_covariance=None,
    ):
        """

        Args:
            y: Shape (..., N, D)
            initialization: Affiliations between 0 and 1. Shape (..., K, N)
            num_classes: Scalar >0
            iterations: Scalar >0
            saliency: Importance weighting for each observation, shape (..., N)
            weight_constant_axis: The axis that is used to calculate the mean
                over the affiliations. The affiliations have the
                shape (..., K, N), so the default value means averaging over
                the sample dimension. Note that averaging over an independent
                axis is supported.
            covariance_type: Either 'full', 'diagonal', or 'spherical'
            fixed_covariance: Learned if None. Otherwise, you need to provide
                the correct shape.

        Returns:
        """
        assert xor(initialization is None, num_classes is None), (
            "Incompatible input combination. "
            "Exactly one of the two inputs has to be None: "
            f"{initialization is None} xor {num_classes is None}"
        )
        assert np.isrealobj(y), y.dtype

        if initialization is None and num_classes is not None:
            *independent, num_observations, _ = y.shape
            affiliation_shape = (*independent, num_classes, num_observations)
            initialization = np.random.uniform(size=affiliation_shape)
            initialization \
                /= np.einsum("...kn->...n", initialization)[..., None, :]

        if saliency is None:
            saliency = np.ones_like(initialization[..., 0, :])

        return self._fit(
            y,
            initialization=initialization,
            iterations=iterations,
            saliency=saliency,
            weight_constant_axis=weight_constant_axis,
            covariance_type=covariance_type,
            fixed_covariance=fixed_covariance,
        )

    def fit_predict(
        self,
        y,
        initialization=None,
        num_classes=None,
        iterations=100,
        *,
        saliency=None,
        weight_constant_axis=(-2,),
        covariance_type="full",
        fixed_covariance=None,
    ):
        """Fit a model. Then just return the posterior affiliations."""
        model = self.fit(
            y=y,
            initialization=initialization,
            num_classes=num_classes,
            iterations=iterations,
            saliency=saliency,
            weight_constant_axis=weight_constant_axis,
            covariance_type=covariance_type,
            fixed_covariance=fixed_covariance,
        )
        return model.predict(y)

    def _fit(
            self,
            y,
            initialization,
            iterations,
            saliency,
            covariance_type,
            fixed_covariance,
            weight_constant_axis,
    ):
        affiliation = initialization  # TODO: Do we need np.copy here?
        model = None
        for iteration in range(iterations):
            if model is not None:
                affiliation = model.predict(y)

            model = self._m_step(
                y,
                affiliation=affiliation,
                saliency=saliency,
                weight_constant_axis=weight_constant_axis,
                covariance_type=covariance_type,
                fixed_covariance=fixed_covariance,
            )

        return model

    def _m_step(
            self,
            x,
            affiliation,
            saliency,
            weight_constant_axis,
            covariance_type,
            fixed_covariance,
    ):
        weight = estimate_mixture_weight(
            affiliation=affiliation,
            saliency=saliency,
            weight_constant_axis=weight_constant_axis,
        )

        gaussian = GaussianTrainer()._fit(
            y=x[..., None, :, :],
            saliency=affiliation * saliency[..., None, :],
            covariance_type=covariance_type
        )

        if fixed_covariance is not None:
            assert fixed_covariance.shape == gaussian.covariance.shape, (
                f'{fixed_covariance.shape} != {gaussian.covariance.shape}'
            )
            gaussian = gaussian.__class__(
                mean=gaussian.mean,
                covariance=fixed_covariance
            )

        return GMM(weight=weight, gaussian=gaussian)


@dataclass
class BinaryGMM(_ProbabilisticModel):
    kmeans: KMeans  # from sklearn

    def predict(self, x):
        """

        Args:
            x: Shape (N, D)

        Returns: Affiliation with shape (K, N)

        """
        N, D = x.shape
        assert np.isrealobj(x), x.dtype

        labels = self.kmeans.predict(x)
        affiliations = labels_to_one_hot(
            labels, self.kmeans.n_clusters, axis=-2, keepdims=False,
            dtype=x.dtype
        )
        assert affiliations.shape == (self.kmeans.n_clusters, N)
        return affiliations


class BinaryGMMTrainer:
    """k-means trainer.
    This is a specific wrapper of sklearn's kmeans for Deep Clustering
    embeddings. This explains the variable names and also the fixed shape for
    the embeddings.
    """
    def fit(
        self,
        x,
        num_classes,
        saliency=None
    ):
        """

        Args:
            x: Shape (N, D)
            num_classes: Scalar >0
            saliency: Importance weighting for each observation, shape (N,)
                Saliency has to be boolean.

        """
        N, D = x.shape
        if saliency is not None:
            assert saliency.dtype == np.bool, (
                'Only boolean saliency supported. '
                f'Current dtype: {saliency.dtype}.'
            )
            assert saliency.shape == (N,)
            x = x[saliency, :]
        return BinaryGMM(kmeans=KMeans(n_clusters=num_classes).fit(x))
