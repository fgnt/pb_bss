from operator import xor

import numpy as np
import scipy.special
from dataclasses import dataclass
from pb_bss.distribution.complex_angular_central_gaussian import (
    ComplexAngularCentralGaussian,
    ComplexAngularCentralGaussianTrainer,
    normalize_observation,
)
from pb_bss.distribution.mixture_model_utils import (
    apply_inline_permutation_alignment,
    estimate_mixture_weight,
    log_pdf_to_affiliation,
)
from pb_bss.distribution.utils import _ProbabilisticModel
from pb_bss.permutation_alignment import _PermutationAlignment

__all__ = [
    'CACGMM',
    'CACGMMTrainer',
    'sample_cacgmm',
    'normalize_observation',
]


def sample_cacgmm(
        size,
        weight,
        covariance,
        return_label=False
):
    assert weight.ndim == 1, weight
    assert isinstance(size, int), size
    assert covariance.ndim == 3, covariance.shape

    num_classes, = weight.shape

    D = covariance.shape[-1]
    assert covariance.shape == (num_classes, D, D), (covariance.shape, num_classes, D)  # noqa

    labels = np.random.choice(range(num_classes), size=size, p=weight)

    x = np.zeros((size, D), dtype=np.complex128)

    for l in range(num_classes):
        cacg = ComplexAngularCentralGaussian.from_covariance(
            covariance=covariance[l, :, :]
        )
        x[labels == l, :] = cacg.sample(size=(np.sum(labels == l),))

    if return_label:
        return x, labels
    else:
        return x


@dataclass
class CACGMM(_ProbabilisticModel):
    weight: np.array  # (..., K, 1) for weight_constant_axis==(-1,)
    cacg: ComplexAngularCentralGaussian

    def predict(self, y, return_quadratic_form=False):
        assert np.iscomplexobj(y), y.dtype
        y = normalize_observation(y)  # swap D and N dim
        affiliation, quadratic_form, _ = self._predict(y)
        if return_quadratic_form:
            return affiliation, quadratic_form
        else:
            return affiliation

    def _predict(self, y, source_activity_mask=None, affiliation_eps=0.):
        """

        Note: y shape is (..., D, N) and not (..., N, D) like in predict

        Args:
            y: Normalized observations with shape (..., D, N).
        Returns: Affiliations with shape (..., K, N) and quadratic format
            with the same shape.

        """
        *independent, _, num_observations = y.shape

        log_pdf, quadratic_form = self.cacg._log_pdf(y[..., None, :, :])

        affiliation = log_pdf_to_affiliation(
            self.weight,
            log_pdf,
            source_activity_mask=source_activity_mask,
            affiliation_eps=affiliation_eps,
        )

        return affiliation, quadratic_form, log_pdf

    def log_likelihood(self, y):
        """

        >>> import paderbox as pb
        >>> F, T, D, K = 513, 400, 6, 3
        >>> y = pb.utils.random_utils.normal([F, T, D], dtype=np.complex128)
        >>> mm = CACGMMTrainer().fit(y, num_classes=K, iterations=2)
        >>> log_likelihood1 = mm.log_likelihood(y)
        >>> mm = CACGMMTrainer().fit(y, initialization=mm, iterations=1)
        >>> log_likelihood2 = mm.log_likelihood(y)
        >>> assert log_likelihood2 > log_likelihood1, (log_likelihood1, log_likelihood2)

        >>> np.isscalar(log_likelihood1), log_likelihood1.dtype
        (True, dtype('float64'))


        """
        assert np.iscomplexobj(y), y.dtype
        y = normalize_observation(y)  # swap D and N dim
        affiliation, quadratic_form, log_pdf = self._predict(y)
        return self._log_likelihood(y, log_pdf)

    def _log_likelihood(self, y, log_pdf):
        """
        Note: y shape is (..., D, N) and not (..., N, D) like in log_likelihood

        Args:
            y: Normalized observations with shape (..., D, N).
            log_pdf: shape (..., K, N)

        Returns:
            log_likelihood, scalar

        """
        *independent, channels, num_observations = y.shape

        # log_pdf.shape: *independent, speakers, num_observations

        # first: sum above the speakers
        # second: sum above time frequency in log domain
        log_likelihood = np.sum(scipy.special.logsumexp(log_pdf, axis=-2))
        return log_likelihood


class CACGMMTrainer:
    def fit(
            self,
            y,
            initialization=None,
            num_classes=None,
            iterations=100,
            *,
            saliency=None,
            source_activity_mask=None,
            weight_constant_axis=(-1,),
            hermitize=True,
            covariance_norm='eigenvalue',
            affiliation_eps=1e-10,
            eigenvalue_floor=1e-10,
            inline_permutation_aligner: _PermutationAlignment = None,
    ):
        """

        Args:
            y: Shape (..., N, D)
            initialization:
                Affiliations between 0 and 1. Shape (..., K, N)
                or CACGMM instance
            num_classes: Scalar >0
            iterations: Scalar >0
            saliency:
                Importance weighting for each observation, shape (..., N)
                Should be pre-calculated externally, not just a string.
            source_activity_mask: Boolean mask that says for each time point
                for each source if it is active or not.
                Shape (..., K, N)
            weight_constant_axis: The axis that is used to calculate the mean
                over the affiliations. The affiliations have the
                shape (..., K, N), so the default value means averaging over
                the sample dimension. Note that averaging over an independent
                axis is supported.
            hermitize:
            covariance_norm: 'eigenvalue', 'trace' or False
            affiliation_eps:
            eigenvalue_floor: Relative flooring of the covariance eigenvalues
            inline_permutation_aligner: In rare cases you may want to run a
                permutation alignment solver after each E-step. You can
                instantiate a permutation alignment solver outside of the
                fit function and pass it to this function.

        Returns:

        """
        assert xor(initialization is None, num_classes is None), (
            "Incompatible input combination. "
            "Exactly one of the two inputs has to be None: "
            f"{initialization is None} xor {num_classes is None}"
        )

        assert np.iscomplexobj(y), y.dtype
        assert y.shape[-1] > 1, y.shape
        y = normalize_observation(y)  # swap D and N dim

        assert iterations > 0, iterations

        model = None

        *independent, D, num_observations = y.shape
        if initialization is None:
            assert num_classes is not None, num_classes
            affiliation_shape = (*independent, num_classes, num_observations)
            affiliation = np.random.uniform(size=affiliation_shape)
            affiliation /= np.einsum("...kn->...n", affiliation)[..., None, :]
            quadratic_form = np.ones(affiliation_shape, dtype=y.real.dtype)
        elif isinstance(initialization, np.ndarray):
            num_classes = initialization.shape[-2]
            assert num_classes > 1, num_classes
            affiliation_shape = (*independent, num_classes, num_observations)

            # Force same number of dims (Prevent wrong input)
            assert initialization.ndim == len(affiliation_shape), (
                initialization.shape, affiliation_shape
            )

            # Allow singleton dimensions to be broadcasted
            assert initialization.shape[-2:] == affiliation_shape[-2:], (
                initialization.shape, affiliation_shape
            )

            affiliation = np.broadcast_to(initialization, affiliation_shape)
            quadratic_form = np.ones(affiliation_shape, dtype=y.real.dtype)
        elif isinstance(initialization, CACGMM):
            # weight[-2] may be 1, when weight is fixed to 1/K
            # num_classes = initialization.weight.shape[-2]
            num_classes = initialization.cacg.covariance_eigenvectors.shape[-3]

            model = initialization
        else:
            raise TypeError('No sufficient initialization.')

        if isinstance(weight_constant_axis, list):
            # List does not work in numpy 1.16.0 as axis
            weight_constant_axis = tuple(weight_constant_axis)

        if source_activity_mask is not None:
            assert source_activity_mask.dtype == np.bool, source_activity_mask.dtype  # noqa
            assert source_activity_mask.shape[-2:] == (num_classes, num_observations), (source_activity_mask.shape, independent, num_classes, num_observations)  # noqa

            if isinstance(initialization, np.ndarray):
                assert source_activity_mask.shape == initialization.shape, (source_activity_mask.shape, initialization.shape)  # noqa

        assert num_classes < 20, f'num_classes: {num_classes}, sure?'
        assert D < 35, f'Channels: {D}, sure?'

        for iteration in range(iterations):
            if model is not None:
                affiliation, quadratic_form, _ = model._predict(
                    y,
                    source_activity_mask=source_activity_mask,
                    affiliation_eps=affiliation_eps,
                )

                if inline_permutation_aligner is not None:
                    affiliation, quadratic_form \
                        = apply_inline_permutation_alignment(
                            affiliation=affiliation,
                            quadratic_form=quadratic_form,
                            weight_constant_axis=weight_constant_axis,
                            aligner=inline_permutation_aligner,
                        )

            model = self._m_step(
                y,
                quadratic_form,
                affiliation=affiliation,
                saliency=saliency,
                hermitize=hermitize,
                covariance_norm=covariance_norm,
                eigenvalue_floor=eigenvalue_floor,
                weight_constant_axis=weight_constant_axis,
            )

        return model

    def fit_predict(
            self,
            y,
            initialization=None,
            num_classes=None,
            iterations=100,
            *,
            saliency=None,
            source_activity_mask=None,
            weight_constant_axis=(-1,),
            hermitize=True,
            covariance_norm='eigenvalue',
            affiliation_eps=1e-10,
            eigenvalue_floor=1e-10,
            inline_permutation_aligner: _PermutationAlignment = None,
    ):
        """Fit a model. Then just return the posterior affiliations."""
        model = self.fit(
            y=y,
            initialization=initialization,
            num_classes=num_classes,
            iterations=iterations,
            saliency=saliency,
            source_activity_mask=source_activity_mask,
            weight_constant_axis=weight_constant_axis,
            hermitize=hermitize,
            covariance_norm=covariance_norm,
            affiliation_eps=affiliation_eps,
            eigenvalue_floor=eigenvalue_floor,
            inline_permutation_aligner=inline_permutation_aligner,
        )
        return model.predict(y)

    def _m_step(
            self,
            x,
            quadratic_form,
            affiliation,
            saliency,
            hermitize,
            covariance_norm,
            eigenvalue_floor,
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

        cacg = ComplexAngularCentralGaussianTrainer()._fit(
            y=x[..., None, :, :],
            saliency=masked_affiliation,
            quadratic_form=quadratic_form,
            hermitize=hermitize,
            covariance_norm=covariance_norm,
            eigenvalue_floor=eigenvalue_floor,
        )
        return CACGMM(weight=weight, cacg=cacg)
