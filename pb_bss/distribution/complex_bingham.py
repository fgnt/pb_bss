from dataclasses import dataclass

from cached_property import cached_property

import numpy as np
from scipy.optimize import least_squares

from pb_bss.distribution.utils import _ProbabilisticModel
from pb_bss.utils import is_broadcast_compatible
from pb_bss.distribution.complex_bingham_utils import grad_log_norm_symbolic, grad_log_norm_symbolic_diff

def normalize_observation(observation):
    """

    Args:
        observation: (..., N, D)

    Returns:
        normalized observation (..., N, D)
    """
    # ToDo: Should the dimensions be swapped like in cacg for speed?
    return observation / np.maximum(
        np.linalg.norm(observation, axis=-1, keepdims=True),
        np.finfo(observation.dtype).tiny,
    )


@dataclass
class ComplexBingham(_ProbabilisticModel):
    covariance_eigenvectors: np.array = None  # (..., D, D)
    covariance_eigenvalues: np.array = None  # (..., D)

    def __post_init__(self):
        self.covariance_eigenvectors = np.array(self.covariance_eigenvectors)
        self.covariance_eigenvalues = np.array(self.covariance_eigenvalues)

    @property
    def covariance(self):
        return np.einsum(
            '...wx,...x,...zx->...wz',
            self.covariance_eigenvectors,
            self.covariance_eigenvalues,
            self.covariance_eigenvectors.conj(),
            optimize='greedy',
        )

    def pdf(self, y):
        """ Calculates pdf function.

        Args:
            y: Assumes shape (..., D).
            loc: Mode vector. Assumes corresponding shape (..., D).
            scale: Concentration parameter with shape (...).

        Returns:
        """
        return np.exp(self.log_pdf(y))

    def log_pdf(self, y):
        """ Calculates logarithm of pdf function.

        Args:
            y: Assumes shape (..., D).
            loc: Mode vector. Assumes corresponding shape (..., D).
            scale: Concentration parameter with shape (...).

        Returns:

        >>> ComplexBingham([[1, 0], [0, 1]], [0.9, 0.1]).log_pdf([[np.sqrt(2), np.sqrt(2)]] * 10)
        array([-1.50913282, -1.50913282, -1.50913282, -1.50913282, -1.50913282,
               -1.50913282, -1.50913282, -1.50913282, -1.50913282, -1.50913282])
        """
        y = np.array(y)

        result = np.einsum("...td,...dD,...tD->...t", y.conj(), self.covariance, y)
        result = result.real
        result -= self.log_norm()[..., None]
        return result

    def log_norm(self, remove_duplicate_eigenvalues=True):
        return np.log(self.norm(remove_duplicate_eigenvalues=remove_duplicate_eigenvalues))

    def norm(self, remove_duplicate_eigenvalues=True, eps=1e-8):
        """
        >>> model = ComplexBingham(None, [0.8       , 0.92679492, 1.27320508])
        >>> model.covariance_eigenvalues
        array([0.8       , 0.92679492, 1.27320508])
        >>> model.norm()
        84.71169626134224
        >>> model = ComplexBingham(None, [0.9, 0.9000000000000001, 1.2])
        >>> model.covariance_eigenvalues
        array([0.9, 0.9, 1.2])

        Numeric problem, because two eigenvalues are equal
        >>> model.norm(remove_duplicate_eigenvalues=False)
        303.2530461789244
        >>> model.norm()
        84.4975422636874

        Stable solution
        >>> ComplexBingham(None, np.array([1, 0.1, 0.1])).norm(remove_duplicate_eigenvalues=True)
        47.34827539909092

        >>> ComplexBingham(None, np.array([1, 0.1+1e-15, 0.1])).norm(remove_duplicate_eigenvalues=False)
        31.006276680299816
        >>> ComplexBingham(None, np.array([1, 0.1+1e-14, 0.1])).norm(remove_duplicate_eigenvalues=False)
        49.41625345922783
        >>> ComplexBingham(None, np.array([1, 0.1+1e-13, 0.1])).norm(remove_duplicate_eigenvalues=False)
        47.35724289842667
        >>> ComplexBingham(None, np.array([1, 0.1+1e-12, 0.1])).norm(remove_duplicate_eigenvalues=False)
        47.34210311489137
        >>> ComplexBingham(None, np.array([1, 0.1+1e-11, 0.1])).norm(remove_duplicate_eigenvalues=False)
        47.349673006659025
        >>> ComplexBingham(None, np.array([1, 0.1+1e-10, 0.1])).norm(remove_duplicate_eigenvalues=False)
        47.34825365195259
        >>> ComplexBingham(None, np.array([1, 0.1+1e-9, 0.1])).norm(remove_duplicate_eigenvalues=False)
        47.3482832218423

        Analytical solution
        >>> 2 * np.pi ** 3 *( np.exp(1) / 0.9**2 -  np.exp(0.1) / 0.9**2 + np.exp(0.1) / (0.1 - 1))
        47.348275222150356

        Independent axis
        >>> ComplexBingham(None, np.array([1, 0.1, 0.1])).norm()
        47.34827539909092
        >>> ComplexBingham(None, np.array([1, 0.1, 0.0])).norm()
        45.92874653819097
        >>> ComplexBingham(None, np.array([[1, 0.1, 0.1], [1, 0.1, 0.0]])).norm()
        array([47.3482754 , 45.92874654])
        >>> ComplexBingham(None, np.array([[0.1, 1, 0.1], [0.1, 1, 0.0]])).norm()
        array([47.3482754 , 45.92874654])

        Higher dimensions
        >>> ComplexBingham(None, np.array([
        ...     5.15996555e-04, 6.28805516e-04, 1.37554184e-03, 1.53621463e-02,
        ...     3.74437619e-02, 9.44673748e-01])).norm()
        19.0955491592929


        >>> values = [-10.00000004, -10.00000003, -10.00000002, -10.00000001, -10., 0.]
        >>> ComplexBingham(None, np.array(values)).norm(eps=1e-8)
        8258270290267.509
        >>> ComplexBingham(None, np.array(values)).norm(eps=1e-7)
        8258270290267.509
        >>> ComplexBingham(None, np.array(values)).norm(eps=1e-6)
        8258270290267.509
        >>> ComplexBingham(None, np.array(values)).norm(eps=1e-5)
        8258270290267.509


        """
        covariance_eigenvalues = self.covariance_eigenvalues
        if remove_duplicate_eigenvalues:
            _, covariance_eigenvalues = self._remove_duplicate_eigenvalues(
                covariance_eigenvalues, eps=eps
            )

        deltas = covariance_eigenvalues[..., None] - covariance_eigenvalues[..., None, :]
        D = deltas.shape[-1]

        deltas[..., range(D), range(D)] = 1
        a = 1 / np.prod(deltas, axis=-1)
        return 2 * np.pi**D * np.sum(a * np.exp(covariance_eigenvalues), axis=-1)

    @classmethod
    def _remove_duplicate_eigenvalues(cls, covariance_eigenvalues, eps=1e-8):
        """
        >>> ComplexBingham._remove_duplicate_eigenvalues(np.array([0.5, 0.5]))[-1]
        array([0.5       , 0.50000001])

        Demonstrate the suboptimal behaviour for duplicate eigenvalues.
        >>> ComplexBingham._remove_duplicate_eigenvalues(np.array([0.2, 0.4, 0.4]), eps=0.02)[-1]
        array([0.2 , 0.4 , 0.42])

        This function sorts the eigenvalues
        >>> ComplexBingham._remove_duplicate_eigenvalues(np.array([0.9, 0.1]))
        (array([1, 0]), array([0.1, 0.9]))
        >>> ComplexBingham._remove_duplicate_eigenvalues(np.array([0.9, 0.06, 0.04]))
        (array([2, 1, 0]), array([0.04, 0.06, 0.9 ]))
        >>> ComplexBingham._remove_duplicate_eigenvalues(np.array([0.9, 0.04, 0.06]))
        (array([2, 0, 1]), array([0.04, 0.06, 0.9 ]))

        >>> ComplexBingham._remove_duplicate_eigenvalues(np.array([1, 0.0, 0.0]))
        (array([2, 0, 1]), array([0.00000000e+00, 1.00000000e-08, 1.00000001e+00]))
        """
        permutation = np.argsort(covariance_eigenvalues, axis=-1, )
        covariance_eigenvalues = np.take_along_axis(covariance_eigenvalues, permutation, axis=-1)
        diff = np.diff(covariance_eigenvalues, axis=-1)
        # eps = covariance_eigenvalues[..., -1] * eps
        # diff = np.maximum(diff, eps[..., None])
        diff = np.maximum(diff, eps)

        # This reconstruction is not optimal, but an error of 1e-8
        covariance_eigenvalues[..., 1:] = (
                covariance_eigenvalues[..., 0][..., None]
                + np.cumsum(diff, axis=-1)
        )

        # https://stackoverflow.com/a/55737198/5766934
        inverse_permutation = np.arange(permutation.shape[-1])[np.argsort(permutation, axis=-1)]
        return inverse_permutation, covariance_eigenvalues


class ComplexBinghamTrainer:
    def __init__(
            self,
            dimension=None,
            max_concentration=np.inf,
            eignevalue_eps=1e-8,
    ):
        """

        Args:
            dimension: Feature dimension. If you do not provide this when
                initializing the trainer, it will be inferred when the fit
                function is called.
        """
        self.dimension = dimension
        assert max_concentration > 0, max_concentration
        self.max_concentration = max_concentration
        self.eignevalue_eps = eignevalue_eps

    @classmethod
    def find_eigenvalues_v2(cls, scatter_eigenvalues, eps=1e-8,
                            max_concentration=np.inf):
        """

        This implementation uses a modified version of the generated files from
        https://github.com/libDirectional/libDirectional/tree/master/lib/util/autogenerated

        ToDo: Generate the source code with python instead of MATLAB.

        >>> ComplexBinghamTrainer.find_eigenvalues_v2([0.9, 0.1])
        array([ 0.        , -9.99544117])
        >>> ComplexBinghamTrainer.find_eigenvalues_v2([0.5, 0.5])
        array([-0.00045475,  0.        ])
        >>> ComplexBinghamTrainer.find_eigenvalues_v2([0.9, 0.06, 0.04])
        array([  0.        , -16.66662429, -24.99999135])
        >>> ComplexBinghamTrainer.find_eigenvalues_v2([0.9, 0.06, 0.03, 0.006, 0.003, 0.001])
        array([   0.        ,  -16.66663119,  -33.33332875, -166.66666412,
               -333.33333091, -999.99999758])

        >>> ComplexBinghamTrainer.find_eigenvalues_v2([
        ...     5.15996555e-04, 6.28805516e-04, 1.37554184e-03, 1.53621463e-02,
        ...     3.74437619e-02, 9.44673748e-01], eps=1e-8)
        array([-1937.99743489, -1590.31683812,  -726.98624711,   -65.09507073,
                 -26.70671827,     0.        ])

        >>> ComplexBinghamTrainer.find_eigenvalues_v2([
        ...     5.15996555e-04, 6.28805516e-04, 1.37554184e-03, 1.53621463e-02,
        ...     3.74437619e-02, 9.44673748e-01], max_concentration=500)
        array([-500.        , -499.99999026, -499.99994411,  -70.4113198 ,
                -27.56045117,    0.        ])
        """
        inverse_permutation, scatter_eigenvalues = ComplexBingham._remove_duplicate_eigenvalues(
            np.array(scatter_eigenvalues), eps=eps
        )

        grad_log_norm_symbolic_d = grad_log_norm_symbolic[scatter_eigenvalues.shape[-1]]

        def foo(x, scatter_eigenvalue):
            ret = grad_log_norm_symbolic_d(*x, 0) - scatter_eigenvalue
            return ret

        x0 = -1 / scatter_eigenvalues
        x0[..., -1] = 0

        if np.isinf(max_concentration):
            pass
        else:
            x0 = np.maximum(
                x0,
                [
                    -(max_concentration - d)
                    for d in range(x0.shape[-1])
                ]
            )
        try:
            res = least_squares(
                foo,
                # Remove the degree of freedom
                x0[..., :-1],
                # The largest eigenvalue is set to one, so all others have to
                # be negative.
                bounds=(-max_concentration, 0),
                kwargs={'scatter_eigenvalue': scatter_eigenvalues}
            )
        except ValueError as e:
            raise ValueError(x0, scatter_eigenvalues) from e

        # Append the dropped eigenvalue
        est = np.array([*res.x, 0])

        # `np.take_along_axis` not necessary, because this function does not
        # support independent axes.
        # np.take_along_axis(covariance_eigenvalues, inverse_permutation,
        #                    axis=-1)
        return est[inverse_permutation]

    @classmethod
    def find_eigenvalues_v3(cls, scatter_eigenvalues, eps=1e-8,
                            max_concentration=np.inf):
        """

        This implementation uses a modified version of the generated files from
        https://github.com/libDirectional/libDirectional/tree/master/lib/util/autogenerated

        ToDo: Generate the source code with python instead of MATLAB.

        >>> ComplexBinghamTrainer.find_eigenvalues_v3([0.9, 0.1])
        array([ 0.        , -9.99544117])
        >>> ComplexBinghamTrainer.find_eigenvalues_v3([0.5, 0.5])
        array([-0.00043799,  0.        ])

        >>> ComplexBinghamTrainer.find_eigenvalues_v3([0.9, 0.06, 0.04])
        array([  0.        , -16.66662429, -24.99999135])

        >>> from pb_bss.distribution.complex_watson import ComplexWatsonTrainer
        >>> t = ComplexWatsonTrainer(dimension=3)
        >>> t.fill_hypergeometric_ratio_inverse([0.9, 0.06, 0.04])
        array([ 19.99999117, -15.51872617, -23.90871118])

        >>> grad_log_norm_symbolic[3](0.        , -16.66662429, -24.99999135)
        [0.8999999999830024, 0.06000000001292287, 0.04000000000407487]
        >>> grad_log_norm_symbolic[3](19.99999117, -15.51872617, -23.90871118)

        >>> ComplexBinghamTrainer.find_eigenvalues_v3([0.9, 0.05, 0.05])
        array([  0.      , -20.      , -19.999996])
        >>> ComplexBinghamTrainer.find_eigenvalues_v3([0.9, 0.0666666667, 0.0333333333])
        array([  0.        , -14.9998715 , -29.99998167])
        >>> t.fill_hypergeometric_ratio_inverse([.9, 0.0666666667, 0.0333333333])
        array([ 19.99999117, -13.83096999, -28.92572221])
        >>> (t.fill_hypergeometric_ratio_inverse([.9, 0.0666666667, 0.0333333333]) - 20)**2
        array([-8.82593048e-06, -3.38309700e+01, -4.89257222e+01])


        >>> t.hypergeometric_ratio(20)
        0.9000000412230742
        >>> grad_log_norm_symbolic[3](20, 1e-8, 1e-9)
        [0.900000041195574, 0.049999878940545125, 0.050000085055916174]

        >>> t = ComplexWatsonTrainer(dimension=2)
        >>> t.fill_hypergeometric_ratio_inverse([0.9, 0.1])
        array([ 9.99544188, -9.99544188])



        >>> ComplexBinghamTrainer.find_eigenvalues_v3([0.9, 0.06, 0.03, 0.006, 0.003, 0.001])
        array([   0.        ,  -16.66663119,  -33.33332875, -166.66666412,
               -333.33333091, -999.9999976 ])

        >>> ComplexBinghamTrainer.find_eigenvalues_v3([
        ...     5.15996555e-04, 6.28805516e-04, 1.37554184e-03, 1.53621463e-02,
        ...     3.74437619e-02, 9.44673748e-01], eps=1e-8)
        array([-1937.99743489, -1590.31683812,  -726.98624711,   -65.09507073,
                 -26.70671827,     0.        ])

        >>> ComplexBinghamTrainer.find_eigenvalues_v3([
        ...     5.15996555e-04, 6.28805516e-04, 1.37554184e-03, 1.53621463e-02,
        ...     3.74437619e-02, 9.44673748e-01], max_concentration=500)
        array([-500.00000002, -500.00000001, -500.        ,  -66.3119293 ,
                -26.90062851,    0.        ])
        """
        inverse_permutation, scatter_eigenvalues = ComplexBingham._remove_duplicate_eigenvalues(
            np.array(scatter_eigenvalues), eps=eps
        )

        grad_log_norm_symbolic_diff_d = grad_log_norm_symbolic_diff[scatter_eigenvalues.shape[-1]]

        def foo(x, scatter_eigenvalue):
            ret = grad_log_norm_symbolic_diff_d(*x, 0) - scatter_eigenvalue
            return ret

        x0 = -1 / scatter_eigenvalues
        x0[..., -1] = 0

        if np.isinf(max_concentration):
            pass
        else:
            x0 = np.maximum(
                x0,
                [
                    -(max_concentration - d)
                    for d in range(x0.shape[-1])
                ]
            )

        # print(x0)
        # Remove the degree of freedom
        # Force order in using the diff
        x0 = -np.diff(x0)
        # print(x0)

        try:
            res = least_squares(
                foo,
                x0,  # the last value was removed in the np.diff
                # The largest eigenvalue is set to one, so all others have to
                # be negative.
                bounds=(-max_concentration, -1e-8),
                kwargs={'scatter_eigenvalue': scatter_eigenvalues}
            )
        except ValueError as e:
            raise ValueError(x0, scatter_eigenvalues) from e

        # Append the dropped eigenvalue
        est = np.cumsum(np.array([*res.x, 0])[..., ::-1])[..., ::-1]

        # `np.take_along_axis` not necessary, because this function does not
        # support independent axes.
        # np.take_along_axis(covariance_eigenvalues, inverse_permutation,
        #                    axis=-1)
        est = est[inverse_permutation]
        if np.isinf(max_concentration):
            return est
        else:
            est = np.maximum(est, -max_concentration)
            inverse_permutation, est = ComplexBingham._remove_duplicate_eigenvalues(
                est, eps=eps
            )
            return est[inverse_permutation]


    def find_eigenvalues_sympy(self, scatter_eigenvalues, start=None):
        """
        ToDo: Get the execution time to a finite value for channels >= 5.

        In the moment this function only supports small number of channels.
        Sympy has some problems when the number of channels gets higher.

        Use find_eigenvalues_v2, that is more stable.

        >>> import sympy
        >>> trainer = ComplexBinghamTrainer(2)
        >>> trainer.find_eigenvalues_sympy([0.9, 0.1])
        array([[ 0.        ],
               [-9.99544094]])

        >>> trainer.grad_log_norm([0.9, 0.1])
        array([0.56596622, 0.43403378])

        >>> trainer.grad_log_norm([0., -9.99544094])
        array([0.9, 0.1])
        >>> trainer.grad_log_norm([0. + 10, -9.99544094 + 10])
        array([0.9, 0.1])

        # >>> ComplexBinghamDistribution.estimateParameterMatrix([0.9, 0, 0; 0, 0.06, 0; 0, 0, 0.04])
        >>> trainer = ComplexBinghamTrainer(3)
        >>> trainer.find_eigenvalues_sympy([0.9, 0.06, 0.04])
        array([[  0.        ],
               [-16.59259207],
               [-24.95061675]])
        >>> trainer.find_eigenvalues_sympy([0.9, 0.05, 0.05])
        array([[  0.        ],
               [-19.93827431],
               [-19.93827213]])

        """
        inverse_permutation, scatter_eigenvalues = ComplexBingham._remove_duplicate_eigenvalues(
            np.array(scatter_eigenvalues)
        )

        import sympy
        out_to_solve = [
            sympy.simplify(o - i)
            for o, i in zip(self.grad_log_norm_symbolic, scatter_eigenvalues)
        ]
        if start is None:
            #  The Complex Bingham Distribution and Shape Analysis
            #  Eq. (3.3)
            start = list(-1 / np.array(scatter_eigenvalues))
        result = sympy.nsolve(
            out_to_solve, self.eigenvalues_symbol, start,
            tol=1e-6)

        res = np.array(result.tolist()).astype(np.float64)
        res = res - np.amax(res)
        return res[inverse_permutation]

    def _doctest_grad_log_norm_symbolic(self):
        """
        >>> import sympy
        >>> trainer = ComplexBinghamTrainer(2)
        >>> trainer.grad_log_norm_symbolic[0]
        ((x0 - x1)*exp(x0) - exp(x0) + exp(x1))/((x0 - x1)*(exp(x0) - exp(x1)))
        >>> trainer.grad_log_norm_symbolic[1]
        (-(x0 - x1)*exp(x1) + exp(x0) - exp(x1))/((x0 - x1)*(exp(x0) - exp(x1)))
        >>> print(sympy.printing.pretty(
        ...     trainer.grad_log_norm_symbolic))  # doctest: +NORMALIZE_WHITESPACE
                    x0    x0    x1               x1    x0    x1
         (x0 - x1)*e   - e   + e    - (x0 - x1)*e   + e   - e
        [-------------------------, ---------------------------]
                     / x0    x1\\                 / x0    x1\\
           (x0 - x1)*\\e   - e  /       (x0 - x1)*\\e   - e  /

        """
    def grad_log_norm(self, eigenvalues):
        subs = dict(zip(self.eigenvalues_symbol, eigenvalues))
        return np.array([
            expr.evalf(subs=subs) for expr in self.grad_log_norm_symbolic
        ]).astype(np.float64)

    @cached_property
    def eigenvalues_symbol(self):
        import sympy
        D = self.dimension
        return sympy.symbols(
            [f'x{d}' for d in range(D)]
        )

    @cached_property
    def grad_log_norm_symbolic(self):
        import sympy
        D = self.dimension
        X = self.eigenvalues_symbol
        B = [1] * D
        for d in range(D):
            for dd in range(D):
                if d != dd:
                    B[d] = B[d] * (X[d] - X[dd])
        B = [1 / b for b in B]

        p_D = sympy.pi ** D

        tmp = [b * sympy.exp(x_) for x_, b in zip(X, B)]
        tmp = sum(tmp)
        symbolic_norm_for_bingham = 2 * p_D * tmp

        return [
            sympy.simplify(sympy.diff(
                sympy.log(symbolic_norm_for_bingham),
                x_
            ))
            for x_ in X
        ]

    def fit(self, y, saliency=None) -> ComplexBingham:
        assert np.iscomplexobj(y), y.dtype
        assert y.shape[-1] > 1
        y = y / np.maximum(
            np.linalg.norm(y, axis=-1, keepdims=True), np.finfo(y.dtype).tiny
        )

        if saliency is not None:
            assert is_broadcast_compatible(y.shape[:-1], saliency.shape), (
                y.shape,
                saliency.shape,
            )

        if self.dimension is None:
            self.dimension = y.shape[-1]
        else:
            assert self.dimension == y.shape[-1], (
                "You initialized the trainer with a different dimension than "
                "you are using to fit a model. Use a new trainer, when you "
                "change the dimension."
            )

        return self._fit(y, saliency=saliency)

    def _fit(self, y, saliency) -> ComplexBingham:
        if saliency is None:
            covariance = np.einsum(
                "...nd,...nD->...dD", y, y.conj()
            )
            denominator = np.array(y.shape[-2])
        else:
            covariance = np.einsum(
                "...n,...nd,...nD->...dD", saliency, y, y.conj()
            )
            denominator = np.einsum("...n->...", saliency)[..., None, None]

        covariance /= denominator
        covariance = force_hermitian(covariance)
        scatter_eigenvalues, eigenvecs = np.linalg.eigh(covariance)

        eigenvalues = np.empty_like(scatter_eigenvalues)
        for index in np.ndindex(scatter_eigenvalues.shape[:-1]):
            assert np.all(scatter_eigenvalues[index] >= 0), scatter_eigenvalues[index]
            eigenvalues[index] = self.find_eigenvalues_v3(
                scatter_eigenvalues[index],
                max_concentration=self.max_concentration,
                eps=self.eignevalue_eps
            )
        return ComplexBingham(
            covariance_eigenvectors=eigenvecs,
            covariance_eigenvalues=eigenvalues,
        )


def force_hermitian(matrix):
    """

    >>> A = np.array([[1+2j, 3+5j], [7+11j, 13+17j]])
    >>> force_hermitian(A)
    array([[ 1.+0.j,  5.-3.j],
           [ 5.+3.j, 13.+0.j]])
    >>> force_hermitian(force_hermitian(A))
    array([[ 1.+0.j,  5.-3.j],
           [ 5.+3.j, 13.+0.j]])
    """
    return (matrix + np.swapaxes(matrix.conj(), -1, -2)) / 2
