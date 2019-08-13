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
    estimate_mixture_weight
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

        # The value of affiliation max exceed float64 range.
        # Scaling (add in log domain) does not change the final affiliation.
        affiliation = log_pdf - np.amax(log_pdf, axis=-2, keepdims=True)

        np.exp(affiliation, out=affiliation)  # inplace

        affiliation *= self.weight

        if source_activity_mask is not None:
            assert source_activity_mask.dtype == np.bool, source_activity_mask.dtype  # noqa
            affiliation *= source_activity_mask

        denominator = np.maximum(
            np.sum(affiliation, axis=-2, keepdims=True),
            np.finfo(affiliation.dtype).tiny,
        )
        affiliation /= denominator

        if affiliation_eps != 0:
            affiliation = np.clip(
                affiliation, affiliation_eps, 1 - affiliation_eps
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
            return_affiliation=False,
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
            source_activity_mask: Boolean mask that says for each time point for
                each source if it is active or not.
                Shape (..., K, N)
            weight_constant_axis: The axis that is used to calculate the mean
                over the affiliations. The affiliations have the
                shape (..., K, N), so the default value means averaging over
                the sample dimension. Note that averaging over an independent
                axis is supported. Averaging over -2 is identical to
                dirichlet_prior_concentration == np.inf.
            hermitize:
            covariance_norm: 'eigenvalue', 'trace' or False
            affiliation_eps:
            eigenvalue_floor: Relative flooring of the covariance eigenvalues
            return_affiliation:
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

        if return_affiliation is True:
            return model, affiliation
        elif return_affiliation is False:
            return model
        else:
            raise ValueError(return_affiliation)

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


# @dataclass
# class ComplexAngularCentralGaussianMixtureModelParameters(_Parameter):
#     cacg: ComplexAngularCentralGaussianParameters = field(
#         default_factory=ComplexAngularCentralGaussianParameters
#     )
#     mixture_weight: np.array = None
#     affiliation: np.array = None
#
#     eps: float = 1e-10
#
#     def _predict(self, Y, source_activity_mask=None):
#         """Predict class affiliation posteriors from given model.
#
#         Args:
#             Y: Normalized observations with shape (..., D, T).
#             source_activity_mask: shape (..., K, T)
#         Returns: Affiliations with shape (..., K, T) and quadratic format
#             with the same shape.
#         """
#         *independent, D, T = Y.shape
#         K = self.mixture_weight.shape[-1]
#
#         quadratic_form = (
#             np.abs(
#                 np.einsum(
#                     "...dt,...kde,...et->...kt",
#                     Y.conj(),
#                     self.cacg.precision,
#                     Y,
#                 )
#             )
#             + self.eps
#         )
#
#         assert quadratic_form.shape == (
#             *independent,
#             K,
#             T,
#         ), quadratic_form.shape
#
#         affiliation = -D * np.log(quadratic_form)
#         affiliation -= np.log(self.cacg.determinant)[..., None]
#         affiliation = np.exp(affiliation)
#         affiliation *= self.mixture_weight[..., None]
#
#         if source_activity_mask is not None:
#             affiliation *= source_activity_mask
#
#         # ToDo: Figure out if
#         #  >>> affiliations /= np.sum(affiliations, axis=-2, keepdims=True) + self.eps
#         # >>> affiliations = np.clip(affiliations, self.eps, 1 - self.eps)
#         # or
#         # >>> affiliations = np.clip(affiliations, self.eps, 1 - self.eps)
#         # >>> affiliations /= np.sum(affiliations, axis=-2, keepdims=True) + self.eps
#         # is better
#
#         affiliation /= np.sum(affiliation, axis=-2, keepdims=True) + self.eps
#         affiliation = np.clip(affiliation, self.eps, 1 - self.eps)
#
#         # if self.visual_debug:
#         #     _plot_affiliations(affiliations)
#         return affiliation, quadratic_form
#
#     def predict(self, Y):
#         """Predict class affiliation posteriors from given model.
#
#         Args:
#             Y: Normalized observations with shape (..., T, D).
#         Returns: Affiliations with shape (..., K, T).
#         """
#         Y2 = np.ascontiguousarray(np.swapaxes(Y, -2, -1))
#         return self._predict(Y2)[0]
#
#
# class ComplexAngularCentralGaussianMixtureModel:
#     """Ito 2016."""
#
#     unit_norm = staticmethod(_unit_norm)
#
#     Parameters = staticmethod(
#         ComplexAngularCentralGaussianMixtureModelParameters
#     )
#
#     def __init__(self, eps=1e-10, visual_debug=False, pbar=False):
#         self.eps = eps
#         self.visual_debug = visual_debug  # ToDo
#         self.pbar = pbar
#
#     def fit(
#         self,
#         Y,
#         initialization,
#         source_activity_mask=None,
#         iterations=100,
#         hermitize=True,
#         trace_norm=True,
#         eigenvalue_floor=1e-10,
#     ) -> ComplexAngularCentralGaussianMixtureModelParameters:
#         """Fit a cACGMM.
#
#         Args:
#             Y: Observations with shape (..., T, D). Do not need to be normalized.
#             initialization: Shape (..., K, T).
#                 affiliation or ComplexAngularCentralGaussianMixtureModelParameters.
#                 Note: this model is special, when affiliation is given,
#                 quadratic_form is initialized as one and the algorithm starts
#                 with the M-step.
#                 When the Parameters (TODO) is given, the algorithm starts with
#                 the E-Step.
#             source_activity_mask: Shape (..., K, T)
#                 A binary mask that indicates if a source is active or not at a
#                 time point. Example about a voice activity detection determines
#                 sections where only noise is active, then this mask can set the
#                 activity of all speakers at that time point to zero.
#
#         The following two examples are equal, both have 20 iterations, but the
#         second splits them in two times 10 iteration:
#
#         >> Model = ComplexAngularCentralGaussianMixtureModel()
#         >> model = Model.fit(Y, init_affiliation, iterations=20)
#
#         >> Model = ComplexAngularCentralGaussianMixtureModel()
#         >> model = Model.fit(Y, init_affiliation, iterations=10)
#         >> model = Model.fit(Y, model, iterations=10)  # ToDo
#         """
#
#         *independent, T, D = Y.shape
#         independent = tuple(independent)
#
#         assert D < 20, (D, "Sure?")
#
#         if isinstance(initialization, self.Parameters):
#             K = initialization.mixture_weight.shape[-1]
#             assert K < 20, (K, "Sure?")
#         else:
#             K = initialization.shape[-2]
#             assert K < 20, (K, "Sure?")
#             assert initialization.shape[-1] == T, (initialization.shape, T)
#             assert initialization.shape[:-2] == independent, (
#                 initialization.shape,
#                 independent,
#             )
#
#         Y = _unit_norm(Y, axis=-1, eps=1e-10, eps_style="where")
#
#         # Y_for_pdf = np.ascontiguousarray(Y)
#         # Y_for_psd = np.ascontiguousarray(np.swapaxes(Y, -2, -1))[..., None, :, :]
#
#         Y_for_pdf = np.ascontiguousarray(np.swapaxes(Y, -2, -1))
#         Y_for_psd = Y_for_pdf[..., None, :, :]
#         # Y_for_psd: Shape (..., 1, T, D)
#
#         if isinstance(initialization, self.Parameters):
#             params = initialization
#         else:
#             params = self.Parameters(eps=self.eps)
#             params.affiliation = np.copy(initialization)  # Shape (..., K, T)
#             quadratic_form = np.ones_like(
#                 params.affiliation
#             )  # Shape (..., K, T)
#
#         # params = ComplexAngularCentralGaussianMixtureModelParameters(
#         #     eps=self.eps
#         # )
#         cacg_model = ComplexAngularCentralGaussian()
#
#         if source_activity_mask is not None:
#             assert (
#                 source_activity_mask.dtype == np.bool
#             ), source_activity_mask.dtype
#             if isinstance(params.affiliation, np.ndarray):
#                 assert (
#                     source_activity_mask.shape == params.affiliation.shape
#                 ), (
#                     source_activity_mask.shape,
#                     params.affiliation.shape,
#                 )
#
#         if isinstance(initialization, self.Parameters):
#             range_iterations = range(1, 1 + iterations)
#         else:
#             range_iterations = range(iterations)
#
#         if self.pbar:
#             import tqdm
#
#             range_iterations = tqdm.tqdm(range_iterations, "cACGMM Iteration")
#         else:
#             range_iterations = range_iterations
#
#         for i in range_iterations:
#             # E step
#             if i > 0:
#                 # Equation 12
#                 del params.affiliation
#                 params.affiliation, quadratic_form = params._predict(
#                     Y_for_pdf, source_activity_mask=source_activity_mask
#                 )
#
#             params.mixture_weight = np.mean(params.affiliation, axis=-1)
#             assert params.mixture_weight.shape == (*independent, K), (
#                 params.mixture_weight.shape,
#                 (*independent, K),
#                 params.affiliation.shape,
#             )
#
#             del params.cacg
#             params.cacg = cacg_model._fit(
#                 Y=Y_for_psd,
#                 saliency=params.affiliation,
#                 quadratic_form=quadratic_form,
#                 hermitize=hermitize,
#                 trace_norm=trace_norm,
#                 eigenvalue_floor=eigenvalue_floor,
#             )
#             del quadratic_form
#         return params
#
#         # if self.visual_debug:
#         #     with context_manager(figure_size=(24, 3)):
#         #         plt.plot(np.log10(
#         #             np.max(eigenvals, axis=-1)
#         #             / np.min(eigenvals, axis=-1)
#         #         ))
#         #         plt.xlabel('frequency bin')
#         #         plt.ylabel('eigenvalue spread')
#         #         plt.show()
#         #     with context_manager(figure_size=(24, 3)):
#         #         plt.plot(self.pi)
#         #         plt.show()
