import numpy as np
from dataclasses import dataclass
from operator import xor
from dc_integration.distribution.gaussian import Gaussian, GaussianTrainer


@dataclass
class GMM:
    weight: np.array = None  # (..., K)
    gaussian: Gaussian = None

    def predict(self, x):
        pass


@dataclass
class WeightFreeGMM:
    gaussian: Gaussian = None

    def predict(self, x):
        pass


class GMMTrainer:
    def __init__(self, eps=1e-10):
        self.eps = eps
        self.log_likelihood_history = []

    def fit(
        self,
        x,
        initialization=None,
        num_classes=None,
        iterations=100,
        saliency=None,
        weight_type="...k",
        covariance_type="full",
    ):
        """

        Args:
            x: Shape (..., N, D)
            initialization: Affiliations between 0 and 1. Shape (..., K, N)
            num_classes: Scalar >0
            iterations: Scalar >0
            saliency: Importance weighting for each observation, shape (..., N)
            weight_type: Either '...k', or None
            covariance_type: Either 'full', 'diagonal', or 'spherical'

        Returns:

        TODO: Support different weight types
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
            initialization /= \
                np.einsum("...kn->...n", initialization)[..., None, :]

        if saliency is None:
            saliency = np.ones_like(initialization[..., 0, :])

        if not weight_type == "...k":
            raise NotImplementedError(weight_type)

        return self._fit(
            x,
            initialization=initialization,
            iterations=iterations,
            saliency=saliency,
            weight_type=weight_type,
            covariance_type=covariance_type,
        )

    def _fit(
        self,
        x,
        initialization,
        iterations,
        saliency,
        weight_type,
        covariance_type,
    ):
        affiliation = initialization  # TODO: Do we need np.copy here?
        for _ in range(iterations):
            weight, gaussian = self._m_step(
                x,
                affiliation=affiliation,
                saliency=saliency,
                weight_type=weight_type,
                covariance_type=covariance_type,
            )
            affiliation = self._e_step(x, weight=weight, gaussian=gaussian)

            # import matplotlib.pyplot as plt
            # f, ax = plt.subplots(1, 1, figsize=(2, 2))
            # ax.scatter(x[:, 0], x[:, 1], c=affiliation[0, :])
            # plt.show()

        return GMM(weight=weight, gaussian=gaussian)

    def _m_step(self, x, affiliation, saliency, weight_type, covariance_type):
        masked_affiliations = affiliation * saliency[..., None, :]
        weight = np.einsum("...kn->...k", masked_affiliations)
        weight /= np.einsum('...n->...', saliency)[..., None]

        gaussian = GaussianTrainer().fit(  # TODO: Change to _fit
            x, masked_affiliations, covariance_type=covariance_type
        )

        return weight, gaussian

    def _e_step(self, x, weight, gaussian):
        # TODO: Can be moved into the parameter class, since it is predict.
        *independent, num_observations, _ = x.shape
        num_classes = weight.shape[-1]
        affiliation_shape = (*independent, num_classes, num_observations)
        affiliation = np.zeros(affiliation_shape)
        affiliation += np.log(weight)[..., :, None]
        affiliation += gaussian.log_pdf(x)
        log_joint_pdf = affiliation
        affiliation = np.exp(affiliation)
        denominator = np.maximum(
            np.einsum("...kn->...n", affiliation)[..., None, :],
            np.finfo(x.dtype).tiny
        )
        affiliation /= denominator

        log_likelihood = np.einsum('...kn,...kn->', affiliation, log_joint_pdf)
        self.log_likelihood_history.append(log_likelihood)
        return affiliation
