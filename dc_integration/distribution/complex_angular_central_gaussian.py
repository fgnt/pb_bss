from dataclasses import dataclass

import numpy as np
from dc_integration.utils import is_broadcast_compatible
from dc_integration.distribution.utils import _Parameter, force_hermitian


@dataclass
class ComplexAngularCentralGaussian:
    covariance: np.array  # (..., D, D)
    eigenvalue_floor: float

    @property
    def determinant_and_precision(self):
        """

        Returns:
            determinant: Determinant of the covariance matrix.
            precision:

        """
        D = self.covariance.shape[-1]
        eigenvals, eigenvecs = np.linalg.eigh(self.covariance)
        eigenvals = eigenvals.real
        eigenvals = np.maximum(
            eigenvals,
            np.max(eigenvals, axis=-1, keepdims=True) * self.eigenvalue_floor,
        )

        # Does not seem to be used anywhere
        # diagonal = np.einsum("de,...d->...de", np.eye(D), eigenvals)
        # covariance = np.einsum(
        #     "...wx,...xy,...zy->...wz", eigenvecs, diagonal, eigenvecs.conj()
        # )

        determinant = np.prod(eigenvals, axis=-1)
        inverse_diagonal = np.einsum(
            "de,...d->...de", np.eye(D), 1 / eigenvals
        )
        precision = np.einsum(
            "...wx,...xy,...zy->...wz",
            eigenvecs,
            inverse_diagonal,
            eigenvecs.conj(),
        )
        return determinant, precision

    def log_pdf(self, x):
        """Gets used by. e.g. the cACGMM.

        Args:
            x: Shape (..., N, D)

        Returns:

        """
        raise NotImplementedError

    def _log_pdf(self, x):
        """
        TODO: quadratic_form might be useful by itself

        Args:
            x: Shape (..., N, D)

        Returns:

        """
        D = x.shape[-1]
        determinant, precision = self.determinant_and_precision
        quadratic_form = np.maximum(
            np.abs(
                np.einsum("...nd,...dD,...nD->...n", x.conj(), precision, x)
            ),
            np.finfo(x.dtype).tiny,
        )

        log_pdf = -D * np.log(quadratic_form)
        log_pdf -= np.log(determinant)[..., None]

        return log_pdf, quadratic_form


class ComplexAngularCentralGaussianTrainer:
    def fit(
        self,
        x,
        saliency=None,
        hermitize=True,
        trace_norm=True,
        eigenvalue_floor=1e-10,
        iterations=10,
    ):
        # raise NotImplementedError(
        #     "needs an iteration for the parameter estimation, because "
        #     "quadratic_form depends on covariance and covariance depends on "
        #     "quadratic_form. "
        #     "Until now nobody needs `fit`, therefore it is not implemented. "
        #     "It should be implemented, since it yields an easy test case."
        # )
        *independent, N, D = x.shape

        if saliency is None:
            quadratic_form = np.ones(*independent, N)
        else:
            raise NotImplementedError

        for _ in range(iterations):
            model = self._fit(
                x=x,
                saliency=saliency,
                quadratic_form=quadratic_form,
                hermitize=hermitize,
                trace_norm=trace_norm,
                eigenvalue_floor=eigenvalue_floor,
            )
            _, quadratic_form = model._log_pdf(x)

        return model

    def _fit(
        self,
        x,
        saliency,
        quadratic_form,
        hermitize=True,
        trace_norm=True,
        eigenvalue_floor=1e-10,
    ) -> ComplexAngularCentralGaussian:
        """Single step of the fit function. In general, needs iterations.

        Args:
            x: Shape (..., N, D)
            saliency: Shape (..., N), e.g. (K, N) for mixture models
            quadratic_form: (..., N), e.g. (K, N) for mixture models
            hermitize:
            trace_norm:
            eigenvalue_floor:

        Returns:

        """
        assert np.iscomplexobj(x), x.dtype

        if saliency is None:
            saliency = np.ones_like(quadratic_form)
        else:
            assert is_broadcast_compatible(x.shape[:-1], saliency.shape), (
                x.shape,
                saliency.shape,
            )

        *independent, N, D = x.shape

        # TODO: I did not understand the need for this yet.
        # independent = list(independent)
        # Special case for mixture model
        # independent[-1] = saliency.shape[-2]
        # K = saliency.shape[-2]

        # TODO: I did not understand the need for this yet.
        # mask = saliency[..., None, :]
        # assert mask.shape == (*independent, 1, T), (
        #     mask.shape,
        #     (*independent, 1, T),
        # )

        covariance = D * np.einsum(
            "...n,...nd,...nD->...dD", (saliency / quadratic_form), x, x.conj()
        )

        if saliency is None:
            denominator = np.array(x.shape[-2])
        else:
            denominator = np.einsum("...n->...", saliency)

        covariance /= denominator
        assert covariance.shape == (*independent, D, D), covariance.shape

        if hermitize:
            covariance = force_hermitian(covariance)

        if trace_norm:
            covariance /= np.einsum("...dd", covariance)[..., None, None]

        return ComplexAngularCentralGaussian(
            covariance=covariance, eigenvalue_floor=eigenvalue_floor
        )


@dataclass
class ComplexAngularCentralGaussianParameters(_Parameter):
    covariance: np.array = None
    precision: np.array = None
    determinant: np.array = None


class ComplexAngularCentralGaussian_deprecated:
    def _fit(
        self,
        Y,
        saliency,
        quadratic_form,
        hermitize=True,
        trace_norm=True,
        eigenvalue_floor=1e-10,
    ) -> ComplexAngularCentralGaussianParameters:
        """
        Attention: Y has swapped dimensions.

        Note:
            `fit` needs an iteration for the parameter estimation, becasue
            quadratic_form depends on covariance and covariance depends on
            quadratic_form.
            Until now nobody need fit, therefore it is not implemented.

        Args:
            Y: Complex observation with shape (..., D, T)
            saliency: Weight for each observation (..., T)

        In case of mixture model:
            Y: Complex observation with shape (..., 1, D, T)
            saliency: Weight for each observation (..., K, T)
        Returns:

        ToDo: Merge ComplexGaussian with ComplexAngularCentralGaussian.
              Both have the same _fit

        """
        assert Y.ndim == saliency.ndim + 1, (Y.shape, saliency.ndim)
        *independent, D, T = Y.shape
        independent = list(independent)

        # Special case for mixture model
        independent[-1] = saliency.shape[-2]
        # K = saliency.shape[-2]

        params = ComplexAngularCentralGaussianParameters()

        mask = saliency[..., None, :]
        assert mask.shape == (*independent, 1, T), (
            mask.shape,
            (*independent, 1, T),
        )

        # params.covariance = D * np.einsum(
        #     '...dt,...et->...de',
        #     (saliency / quadratic_form)[..., None, :] * Y,
        #     Y.conj()
        # )
        params.covariance = D * np.einsum(
            "...t,...dt,...et->...de", (saliency / quadratic_form), Y, Y.conj()
        )

        normalization = np.sum(mask, axis=-1, keepdims=True)
        params.covariance /= normalization
        assert params.covariance.shape == (
            *independent,
            D,
            D,
        ), params.covariance.shape

        if hermitize:
            params.covariance = force_hermitian(params.covariance)

        if trace_norm:
            params.covariance /= np.einsum("...dd", params.covariance)[
                ..., None, None
            ]

        eigenvals, eigenvecs = np.linalg.eigh(params.covariance)
        eigenvals = eigenvals.real
        eigenvals = np.maximum(
            eigenvals,
            np.max(eigenvals, axis=-1, keepdims=True) * eigenvalue_floor,
        )
        diagonal = np.einsum("de,...d->...de", np.eye(D), eigenvals)
        params.covariance = np.einsum(
            "...wx,...xy,...zy->...wz", eigenvecs, diagonal, eigenvecs.conj()
        )
        params.determinant = np.prod(eigenvals, axis=-1)
        inverse_diagonal = np.einsum(
            "de,...d->...de", np.eye(D), 1 / eigenvals
        )
        params.precision = np.einsum(
            "...wx,...xy,...zy->...wz",
            eigenvecs,
            inverse_diagonal,
            eigenvecs.conj(),
        )

        return params
