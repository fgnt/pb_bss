import numpy as np
from numpy.testing import assert_allclose
import unittest
from pb_bss.distribution import CACGMMTrainer
from pb_bss.distribution import ComplexAngularCentralGaussian
import itertools


class TestCACGMM(unittest.TestCase):
    def test_cacgmm(self):
        samples = 10000
        weight = np.array([0.3, 0.7])
        num_classes = weight.shape[0]
        labels = np.random.choice(
            range(num_classes), size=(samples,), p=weight
        )
        covariance = np.array(
            [
                [[10, 1 + 1j, 1 + 1j], [1 - 1j, 5, 1], [1 - 1j, 1, 2]],
                [[2, 0, 0], [0, 3, 0], [0, 0, 2]],
            ]
        )
        covariance /= np.trace(covariance, axis1=-2, axis2=-1)[..., None, None]
        dimension = covariance.shape[-1]
        x = np.zeros((samples, dimension), dtype=np.complex128)

        for l in range(num_classes):
            cacg = ComplexAngularCentralGaussian.from_covariance(
                covariance=covariance[l, :, :]
            )
            x[labels == l, :] = cacg.sample(size=(np.sum(labels == l),))

        model = CACGMMTrainer().fit(
            x,
            num_classes=2,
            covariance_norm='trace',
        )

        # Permutation invariant testing
        permutations = list(itertools.permutations(range(2)))
        best_permutation, best_cost = None, np.inf
        for p in permutations:
            cost = np.linalg.norm(model.cacg.covariance[p, :] - covariance)
            if cost < best_cost:
                best_permutation, best_cost = p, cost

        assert_allclose(
            model.cacg.covariance[best_permutation, :], covariance, atol=0.1
        )

        model.weight = model.weight[best_permutation,]
        assert model.weight[0] < model.weight[1], model.weight
        assert_allclose(model.weight, weight, atol=0.15)

        model = CACGMMTrainer().fit(
            x,
            num_classes=2,
            covariance_norm='trace',
            dirichlet_prior_concentration=np.inf
        )
        assert_allclose(model.weight, [0.5, 0.5])

        model = CACGMMTrainer().fit(
            x,
            num_classes=2,
            covariance_norm='trace',
            dirichlet_prior_concentration=1_000_000_000
        )
        assert_allclose(model.weight, [0.5, 0.5])

    def test_cacgmm_independent_dimension(self):
        samples = 10000
        weight = np.array([0.3, 0.7])
        num_classes = weight.shape[0]
        labels = np.random.choice(
            range(num_classes), size=(samples,), p=weight
        )
        covariance = np.array(
            [
                [[10, 1 + 1j, 1 + 1j], [1 - 1j, 5, 1], [1 - 1j, 1, 2]],
                [[2, 0, 0], [0, 3, 0], [0, 0, 2]],
            ]
        )
        covariance /= np.trace(covariance, axis1=-2, axis2=-1)[..., None, None]
        dimension = covariance.shape[-1]
        x = np.zeros((samples, dimension), dtype=np.complex128)

        for l in range(num_classes):
            cacg = ComplexAngularCentralGaussian.from_covariance(
                covariance=covariance[l, :, :]
            )
            x[labels == l, :] = cacg.sample(size=(np.sum(labels == l),))

        model = CACGMMTrainer().fit(
            x[None, ...],
            num_classes=2,
            covariance_norm='trace',
        )

        # Permutation invariant testing
        permutations = list(itertools.permutations(range(2)))
        best_permutation, best_cost = None, np.inf
        for p in permutations:
            cost = np.linalg.norm(model.cacg.covariance[0, p, :] - covariance)
            if cost < best_cost:
                best_permutation, best_cost = p, cost

        assert_allclose(
            model.cacg.covariance[0, best_permutation, :], covariance, atol=0.1
        )
