import numpy as np
from numpy.testing import assert_allclose
import unittest
from dc_integration.distribution import GMMTrainer
import itertools


class TestGMM(unittest.TestCase):
    def test_gmm(self):
        samples = 1000
        weight = np.array([0.3, 0.7])
        num_classes = weight.shape[0]
        labels = np.random.choice(range(num_classes), size=(samples,), p=weight)
        mean = np.array([[-1, -1], [1, 1]])
        dimension = mean.shape[-1]
        covariance = np.tile(0.25 * np.eye(dimension), (2, 1, 1))

        x = np.zeros((samples, dimension))
        x[labels == 0, :] = np.random.multivariate_normal(
            mean[0, :], covariance[0, :, :], size=(np.sum(labels == 0),)
        )
        x[labels == 1, :] = np.random.multivariate_normal(
            mean[1, :], covariance[0, :, :], size=(np.sum(labels == 1),)
        )

        model = GMMTrainer().fit(x, num_classes=2)

        # Permutation invariant testing
        permutations = list(itertools.permutations(range(2)))
        best_permutation, best_cost = None, np.inf
        for p in permutations:
            cost = np.sum((model.gaussian.mean[p, :] - mean) ** 2)
            if cost < best_cost:
                best_permutation, best_cost = p, cost

        assert_allclose(
            model.gaussian.mean[best_permutation, :], mean, atol=0.2
        )
        assert_allclose(
            model.gaussian.covariance[best_permutation, :], covariance, atol=0.2
        )

    def test_gmm_independent_dimension(self):
        samples = 1000
        weight = np.array([0.3, 0.7])
        num_classes = weight.shape[0]
        labels = np.random.choice(range(num_classes), size=(samples,), p=weight)
        mean = np.array([[-1, -1], [1, 1]])
        dimension = mean.shape[-1]
        covariance = np.tile(0.25 * np.eye(dimension), (2, 1, 1))

        x = np.zeros((samples, dimension))
        x[labels == 0, :] = np.random.multivariate_normal(
            mean[0, :], covariance[0, :, :], size=(np.sum(labels == 0),)
        )
        x[labels == 1, :] = np.random.multivariate_normal(
            mean[1, :], covariance[0, :, :], size=(np.sum(labels == 1),)
        )

        model = GMMTrainer().fit(x[None, ...], num_classes=2)

        # Permutation invariant testing
        permutations = list(itertools.permutations(range(2)))
        best_permutation, best_cost = None, np.inf
        for p in permutations:
            cost = np.sum((model.gaussian.mean[0, p, :] - mean) ** 2)
            if cost < best_cost:
                best_permutation, best_cost = p, cost

        assert_allclose(
            model.gaussian.mean[0, best_permutation, :], mean, atol=0.2
        )
        assert_allclose(
            model.gaussian.covariance[0, best_permutation, :], covariance, atol=0.2
        )
