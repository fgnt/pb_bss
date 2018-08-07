"""
References:
    David E. Tyler
    Statistical analysis for the angular central Gaussian distribution on the
    sphere
    1987

    N. Ito, S. Araki, T. Nakatani
    Complex angular central Gaussian mixture model for directional statistics in
    mask-based microphone array signal processing
    2016
    https://www.eurasip.org/Proceedings/Eusipco/Eusipco2016/papers/1570256519.pdf
"""

from dataclasses import dataclass

import numpy as np
from dc_integration.utils import is_broadcast_compatible
from dc_integration.distribution.utils import force_hermitian
from dc_integration.distribution.utils import _ProbabilisticModel
from dc_integration.distribution.circular_symmetric_gaussian import (
    CircularSymmetricGaussian
)


@dataclass
class ComplexAngularCentralGaussian(_ProbabilisticModel):
    covariance: np.array  # (..., D, D)
    eigenvalue_floor: float = None

    def sample(self, size):
        csg = CircularSymmetricGaussian(covariance=self.covariance)
        x = csg.sample(size=size)
        x /= np.linalg.norm(x, axis=-1, keepdims=True)
        return x

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

        # TODO: Do we always want (automatic) flooring?
        if self.eigenvalue_floor is not None:
            eigenvals = np.maximum(
                eigenvals,
                np.max(eigenvals, axis=-1, keepdims=True)
                * self.eigenvalue_floor,
            )

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
        """

        Args:
            x: Shape (..., N, D)

        Returns:

        """
        x = x / np.maximum(
            np.linalg.norm(x, axis=-1, keepdims=True), np.finfo(x.dtype).tiny
        )
        log_pdf, _ = self._log_pdf(x)
        return log_pdf

    def _log_pdf(self, x):
        """Gets used by. e.g. the cACGMM.
        TODO: quadratic_form might be useful by itself

        Args:
            x: Shape (..., N, D)

        Returns:

        """
        assert is_broadcast_compatible(
            x.shape[:-2], self.covariance.shape[:-2]
        ), (x.shape, self.covariance.shape)

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
        """

        Args:
            x: Should be normalized to unit norm. We normalize it anyway again.
            saliency:
            hermitize:
            trace_norm:
            eigenvalue_floor:
            iterations:

        Returns:

        """
        *independent, N, D = x.shape
        x = x / np.maximum(
            np.linalg.norm(x, axis=-1, keepdims=True), np.finfo(x.dtype).tiny
        )

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

        assert is_broadcast_compatible(
            x.shape[:-2], quadratic_form.shape[:-1]
        ), (x.shape, quadratic_form.shape)
        assert is_broadcast_compatible(
            x.shape[:-2], quadratic_form.shape[:-1]
        ), (x.shape, quadratic_form.shape)
        assert is_broadcast_compatible(
            x.shape[:-2], quadratic_form.shape[:-1]
        ), (x.shape, quadratic_form.shape)

        D = x.shape[-1]
        *independent, N = quadratic_form.shape

        if saliency is None:
            saliency = 1
            denominator = np.array(x.shape[-2])
        else:
            denominator = np.einsum("...n->...", saliency)[..., None, None]

        covariance = D * np.einsum(
            "...n,...nd,...nD->...dD", (saliency / quadratic_form), x, x.conj()
        )
        covariance /= denominator
        assert covariance.shape == (*independent, D, D), covariance.shape

        if hermitize:
            covariance = force_hermitian(covariance)

        if trace_norm:
            covariance /= np.einsum("...dd", covariance)[..., None, None]

        return ComplexAngularCentralGaussian(
            covariance=covariance, eigenvalue_floor=eigenvalue_floor
        )
