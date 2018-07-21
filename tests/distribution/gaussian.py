import numpy as np
from numpy.testing import assert_allclose
import unittest
from dc_integration.distribution import GaussianTrainer


class TestGaussian(unittest.TestCase):
    def test_gaussian(self):
        samples = 10000
        mean = np.ones((3,))
        covariance = 2 * np.eye(3)
        x = np.random.multivariate_normal(mean, covariance, size=(samples,))
        model = GaussianTrainer().fit(x)
        assert_allclose(model.mean, mean, atol=0.1)
        assert_allclose(model.covariance, covariance, atol=0.1)

    def test_diagonal_gaussian(self):
        samples = 10000
        mean = np.ones((3,))
        covariance = 2 * np.eye(3)
        x = np.random.multivariate_normal(mean, covariance, size=(samples,))
        model = GaussianTrainer().fit(x, covariance_type="diagonal")
        assert_allclose(model.mean, mean, atol=0.1)
        assert_allclose(model.covariance, np.diag(covariance), atol=0.1)

    def test_spherical_gaussian(self):
        samples = 10000
        mean = np.ones((3,))
        covariance = 2 * np.eye(3)
        x = np.random.multivariate_normal(mean, covariance, size=(samples,))
        model = GaussianTrainer().fit(x, covariance_type="spherical")
        assert_allclose(model.mean, mean, atol=0.1)
        assert_allclose(
            model.covariance, np.mean(np.diag(covariance)), atol=0.1
        )
