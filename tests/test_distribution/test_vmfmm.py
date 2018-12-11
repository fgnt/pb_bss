import numpy as np
from numpy.testing import assert_equal
import unittest
from pb_bss.distribution import VMFMMTrainer
import itertools


class TestGMM(unittest.TestCase):
    def test_vmfmm_shapes(self):
        # TODO: Sample from the correct distribution please
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

        model = VMFMMTrainer().fit(x, num_classes=2)

        # Permutation invariant testing
        permutations = list(itertools.permutations(range(2)))
        best_permutation, best_cost = None, np.inf
        for p in permutations:
            cost = np.sum((model.vmf.mean[p, :] - mean) ** 2)
            if cost < best_cost:
                best_permutation, best_cost = p, cost

        assert_equal(model.vmf.mean.shape, mean.shape)
        assert_equal(model.vmf.concentration.shape, (num_classes,))
