from dataclasses import dataclass, field

import numpy as np
from dc_integration.distribution.complex_angular_central_gaussian import (
    ComplexAngularCentralGaussianParameters,
    ComplexAngularCentralGaussian,
)
from dc_integration.distribution.utils import (
    _unit_norm,
    _Parameter,
)


@dataclass
class ComplexAngularCentralGaussianMixtureModelParameters(_Parameter):
    cacg: ComplexAngularCentralGaussianParameters \
        = field(default_factory=ComplexAngularCentralGaussianParameters)
    mixture_weight: np.array = None
    affiliation: np.array = None

    eps: float = 1e-10

    def _predict(self, Y, source_activity_mask=None):
        """Predict class affiliation posteriors from given model.

        Args:
            Y: Normalized observations with shape (..., D, T).
            source_activity_mask: shape (..., K, T)
        Returns: Affiliations with shape (..., K, T) and quadratic format
            with the same shape.
        """
        *independent, D, T = Y.shape
        K = self.mixture_weight.shape[-1]

        quadratic_form = np.abs(
            np.einsum(
                '...dt,...kde,...et->...kt',
                Y.conj(),
                self.cacg.precision,
                Y,
            )
        ) + self.eps

        assert quadratic_form.shape == (*independent, K, T), quadratic_form.shape

        affiliation = - D * np.log(quadratic_form)
        affiliation -= np.log(self.cacg.determinant)[..., None]
        affiliation = np.exp(affiliation)
        affiliation *= self.mixture_weight[..., None]

        if source_activity_mask is not None:
            affiliation *= source_activity_mask

        # ToDo: Figure out if
        #  >>> affiliations /= np.sum(affiliations, axis=-2, keepdims=True) + self.eps
        # >>> affiliations = np.clip(affiliations, self.eps, 1 - self.eps)
        # or
        # >>> affiliations = np.clip(affiliations, self.eps, 1 - self.eps)
        # >>> affiliations /= np.sum(affiliations, axis=-2, keepdims=True) + self.eps
        # is better

        affiliation /= np.sum(affiliation, axis=-2, keepdims=True) + self.eps
        affiliation = np.clip(affiliation, self.eps, 1 - self.eps)

        # if self.visual_debug:
        #     _plot_affiliations(affiliations)
        return affiliation, quadratic_form

    def predict(self, Y):
        """Predict class affiliation posteriors from given model.

        Args:
            Y: Normalized observations with shape (..., T, D).
        Returns: Affiliations with shape (..., K, T).
        """
        Y2 = np.ascontiguousarray(np.swapaxes(Y, -2, -1))
        return self._predict(Y2)[0]


class ComplexAngularCentralGaussianMixtureModel:
    """Ito 2016."""
    unit_norm = staticmethod(_unit_norm)

    Parameters = staticmethod(ComplexAngularCentralGaussianMixtureModelParameters)

    def __init__(self, eps=1e-10, visual_debug=False, pbar=False):
        self.eps = eps
        self.visual_debug = visual_debug  # ToDo
        self.pbar = pbar

    def fit(
            self,
            Y,
            initialization,
            source_activity_mask=None,
            iterations=100,
            hermitize=True,
            trace_norm=True,
            eigenvalue_floor=1e-10,
    ) -> ComplexAngularCentralGaussianMixtureModelParameters:
        """Fit a cACGMM.

        Args:
            Y: Observations with shape (..., T, D). Do not need to be normalized.
            initialization: Shape (..., K, T).
                affiliation or ComplexAngularCentralGaussianMixtureModelParameters.
                Note: this model is special, when affiliation is given,
                quadratic_form is initialized as one and the algoriithm starts
                with the M-step.
                When the Parameters (TODO) is given, the algorithm starts with
                the E-Step.
            source_activity_mask: Shape (..., K, T)
                A binary mask that indicates if a source is active or not at a
                time point. Example about a voice activity detection determines
                sections where only noise is active, then this mask can set the
                activity of all speakers at that time point to zero.

        The following two examples are equal, both have 20 iterations, but the
        second splits them in two times 10 iteration:

        >> Model = ComplexAngularCentralGaussianMixtureModel()
        >> model = Model.fit(Y, init_affiliation, iterations=20)

        >> Model = ComplexAngularCentralGaussianMixtureModel()
        >> model = Model.fit(Y, init_affiliation, iterations=10)
        >> model = Model.fit(Y, model, iterations=10)  # ToDo
        """

        *independent, T, D = Y.shape
        independent = tuple(independent)

        assert D < 20, (D, 'Sure?')

        if isinstance(initialization, self.Parameters):
            K = initialization.mixture_weight.shape[-1]
            assert K < 20, (K, 'Sure?')
        else:
            K = initialization.shape[-2]
            assert K < 20, (K, 'Sure?')
            assert initialization.shape[-1] == T, (initialization.shape, T)
            assert initialization.shape[:-2] == independent, (initialization.shape, independent)

        Y = _unit_norm(
            Y,
            axis=-1,
            eps=1e-10,
            eps_style='where'
        )

        # Y_for_pdf = np.ascontiguousarray(Y)
        # Y_for_psd = np.ascontiguousarray(np.swapaxes(Y, -2, -1))[..., None, :, :]

        Y_for_pdf = np.ascontiguousarray(np.swapaxes(Y, -2, -1))
        Y_for_psd = Y_for_pdf[..., None, :, :]
        # Y_for_psd: Shape (..., 1, T, D)

        if isinstance(initialization, self.Parameters):
            params = initialization
        else:
            params = self.Parameters(eps=self.eps)
            params.affiliation = np.copy(initialization)  # Shape (..., K, T)
            quadratic_form = np.ones_like(params.affiliation)  # Shape (..., K, T)

        # params = ComplexAngularCentralGaussianMixtureModelParameters(
        #     eps=self.eps
        # )
        cacg_model = ComplexAngularCentralGaussian()

        if source_activity_mask is not None:
            assert source_activity_mask.dtype == np.bool, source_activity_mask.dtype
            if isinstance(params.affiliation, np.ndarray):
                assert source_activity_mask.shape == params.affiliation.shape, (source_activity_mask.shape, params.affiliation.shape)

        if isinstance(initialization, self.Parameters):
            range_iterations = range(1, 1+iterations)
        else:
            range_iterations = range(iterations)

        if self.pbar:
            import tqdm
            range_iterations = tqdm.tqdm(range_iterations, 'cACGMM Iteration')
        else:
            range_iterations = range_iterations

        for i in range_iterations:
            # E step
            if i > 0:
                # Equation 12
                del params.affiliation
                params.affiliation, quadratic_form = params._predict(
                    Y_for_pdf,
                    source_activity_mask=source_activity_mask,
                )

            params.mixture_weight = np.mean(params.affiliation, axis=-1)
            assert params.mixture_weight.shape == (*independent, K), (params.mixture_weight.shape, (*independent, K), params.affiliation.shape)

            del params.cacg
            params.cacg = cacg_model._fit(
                Y=Y_for_psd,
                saliency=params.affiliation,
                quadratic_form=quadratic_form,
                hermitize=hermitize,
                trace_norm=trace_norm,
                eigenvalue_floor=eigenvalue_floor,
            )
            del quadratic_form
        return params

            # if self.visual_debug:
            #     with context_manager(figure_size=(24, 3)):
            #         plt.plot(np.log10(
            #             np.max(eigenvals, axis=-1)
            #             / np.min(eigenvals, axis=-1)
            #         ))
            #         plt.xlabel('frequency bin')
            #         plt.ylabel('eigenvalue spread')
            #         plt.show()
            #     with context_manager(figure_size=(24, 3)):
            #         plt.plot(self.pi)
            #         plt.show()
