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
    stable: bool = False

    def _predict(self, Y, source_activity_mask=None):
        """Predict class affiliation posteriors from given model.

        Args:
            Y: Normalized observations with shape (..., D, T).
            source_activity_mask: shape (..., K, T)
        Returns: Affiliations with shape (..., K, T) and quadratic format
            with the same shape.

        >>> params = ComplexAngularCentralGaussianMixtureModelParameters()
        >>> params.cacg.precision = np.array([[[1, 0.1], [0.1, 0.5]], [[0.5, 0.1], [0.1, 1]]])
        >>> params.cacg.log_determinant = np.linalg.slogdet(np.linalg.inv(params.cacg.precision))[1]
        >>> params.mixture_weight = np.array([0.6, 0.4])
        >>> Y = np.array([[1, 0], [0.5, 0.6], [0, 1]])
        >>> params.predict(Y)
        array([[0.27272727, 0.64981495, 0.85714286],
               [0.72727273, 0.35018505, 0.14285714]])
        >>> params.stable = True
        >>> params.predict(Y)
        array([[0.27272727, 0.64981495, 0.85714286],
               [0.72727273, 0.35018505, 0.14285714]])

        """

        *independent, D, T = Y.shape
        K = self.mixture_weight.shape[-1]

        # Y: ..., D, T
        # self.cacg.covariance_eigenvectors: (..., K, D, D)
        # self.cacg.covariance_eigenvalues: (..., K, D)
        # precision: covariance_eigenvectors @ covariance_eigenvalues @ covariance_eigenvectors

        # tmp = np.einsum(
        #     '...dt,...kde->...ket',
        #     Y.conj(),
        #     self.cacg.covariance_eigenvectors,
        # )
        quadratic_form = np.maximum(
            np.abs(
                np.einsum(
                    '...dt,...kde,...ke,...kge,...gt->...kt',
                    Y.conj(),
                    self.cacg.covariance_eigenvectors,
                    1 / self.cacg.covariance_eigenvalues,
                    self.cacg.covariance_eigenvectors.conj(),
                    Y,
                    optimize='optimal',
                )
            ),
            np.finfo(Y.dtype).tiny,
        )

        # quadratic_form2 = np.abs(
        #     np.einsum(
        #         '...dt,...kde,...et->...kt',
        #         Y.conj(),
        #         self.cacg.precision,
        #         Y,
        #         optimize='greedy',
        #     )
        # )# + self.eps

        # import cbj
        # cbj.testing.assert_allclose(quadratic_form2, quadratic_form, rtol=1e-10, atol=1e-10)
        #
        # quadratic_form = quadratic_form2 + self.eps


        # np.squeeze(np.swapaxes(self.cacg.precision @ Y[..., None, :, :], -2, -1)[..., None, :] @ np.swapaxes(Y, -1, -2)[..., None, :, :, None], (-1, -2))

        assert quadratic_form.shape == (*independent, K, T), quadratic_form.shape

        affiliation = - D * np.log(quadratic_form)
        # affiliation -= np.log(self.cacg.determinant)[..., None]
        affiliation -= self.cacg.log_determinant[..., None]
        affiliation += np.log(self.mixture_weight)[..., :, None]
        if self.stable:
            affiliation -= np.amax(affiliation, axis=-2, keepdims=True)

        affiliation = np.exp(affiliation)
        # affiliation *= self.mixture_weight[..., None]
        if source_activity_mask is not None:
            affiliation *= source_activity_mask

        # ToDo: Figure out if
        #  >>> affiliations /= np.sum(affiliations, axis=-2, keepdims=True) + self.eps
        # >>> affiliations = np.clip(affiliations, self.eps, 1 - self.eps)
        # or
        # >>> affiliations = np.clip(affiliations, self.eps, 1 - self.eps)
        # >>> affiliations /= np.sum(affiliations, axis=-2, keepdims=True) + self.eps
        # is better
        # assert np.all(np.isfinite(affiliation))

        affiliation /= np.maximum(
            np.sum(affiliation, axis=-2, keepdims=True),
            np.finfo(affiliation.dtype).tiny,
        )  # + self.eps
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

    def __init__(
            self,
            eps=1e-10,
            use_pinv=False,
            visual_debug=False,
            pbar=False,
            stable=True,
    ):
        self.eps = eps
        self.visual_debug = visual_debug  # ToDo
        self.pbar = pbar
        self.use_pinv = use_pinv
        self.stable = stable

    def fit(
            self,
            Y,
            initialization,
            source_activity_mask=None,
            iterations=100,
            saliency=None,
            hermitize=True,
            trace_norm=True,
            eigenvalue_floor=1e-10,
            dirichlet_prior_concentration=1,
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

        >> Trainer.custom_fit

        """
        *independent, T, D = Y.shape
        independent = tuple(independent)

        assert D < 30, (D, 'Sure?')

        if isinstance(initialization, self.Parameters):
            K = initialization.mixture_weight.shape[-1]
            assert K < 20, (K, 'Sure?')
        else:
            K = initialization.shape[-2]
            assert K < 20, (K, 'Sure?')
            assert initialization.shape[-1] == T, (initialization.shape, T)
            assert initialization.shape[:-2] == independent, (initialization.shape, independent)

        if isinstance(saliency, str):
            if saliency == 'norm':
                saliency = np.linalg.norm(Y)
            else:
                raise NotImplementedError(saliency)

        Y = _unit_norm(
            Y,
            axis=-1,
            eps=1e-10,
            eps_style='where',
        )

        # Y_for_pdf = np.ascontiguousarray(Y)
        # Y_for_psd = np.ascontiguousarray(np.swapaxes(Y, -2, -1))[..., None, :, :]

        Y_for_pdf = np.ascontiguousarray(np.swapaxes(Y, -2, -1))
        Y_for_psd = Y_for_pdf[..., None, :, :]
        # Y_for_psd: Shape (..., 1, T, D)

        if isinstance(initialization, self.Parameters):
            params = initialization
            params.stable = self.stable
        else:
            params = self.Parameters(eps=self.eps, stable=self.stable)
            params.affiliation = np.copy(initialization)  # Shape (..., K, T)
            quadratic_form = np.ones_like(params.affiliation)  # Shape (..., K, T)

        # params = ComplexAngularCentralGaussianMixtureModelParameters(
        #     eps=self.eps
        # )
        cacg_model = ComplexAngularCentralGaussian(use_pinv=self.use_pinv)

        if source_activity_mask is not None:
            assert source_activity_mask.dtype == np.bool, source_activity_mask.dtype
            assert source_activity_mask.shape[-2:] == (K, T), (source_activity_mask.shape, independent, K, T)

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
                # old_affiliation = params.affiliation
                params.affiliation, quadratic_form = params._predict(
                    Y_for_pdf,
                    source_activity_mask=source_activity_mask,
                )
                # avg_change = np.mean(np.abs(params.affiliation - old_affiliation))
                # print('Converged', avg_change)

            affiliation = params.affiliation

            if saliency is None:
                if dirichlet_prior_concentration == 1:
                    params.mixture_weight = np.mean(params.affiliation, axis=-1)
                elif np.isposinf(dirichlet_prior_concentration):
                    K, T = params.affiliation.shape[-2:]
                    params.mixture_weight = np.broadcast_to(1 / K, params.affiliation.shape[:-1])
                else:
                    assert dirichlet_prior_concentration >= 1, dirichlet_prior_concentration
                    # params.affiliation: ..., K, T
                    tmp = np.sum(params.affiliation, axis=-1)
                    K, T = params.affiliation.shape[-2:]

                    params.mixture_weight = (
                        tmp + (dirichlet_prior_concentration - 1)
                    ) / (
                        T + (dirichlet_prior_concentration - 1) * K
                    )
            else:
                assert dirichlet_prior_concentration is None, dirichlet_prior_concentration
                affiliation = affiliation * saliency
                params.mixture_weight = _unit_norm(
                    np.sum(affiliation, axis=-1),
                    ord=1,
                    axis=-1,
                    eps=1e-10,
                    eps_style='where',
                )
            assert params.mixture_weight.shape == (*independent, K), (params.mixture_weight.shape, (*independent, K), params.affiliation.shape)

            del params.cacg
            params.cacg = cacg_model._fit(
                Y=Y_for_psd,
                saliency=affiliation,
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
