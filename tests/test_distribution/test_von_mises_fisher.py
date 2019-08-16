import numpy as np
from numpy.testing import assert_allclose, assert_equal
import unittest
from pb_bss.distribution import VonMisesFisher
from pb_bss.distribution import VonMisesFisherTrainer


class TestGaussian(unittest.TestCase):
    def test_shapes(self):
        samples = 10000
        mean = np.ones((3,))
        covariance = np.eye(3)
        x = np.random.multivariate_normal(mean, covariance, size=(samples,))
        model = VonMisesFisherTrainer().fit(x)
        assert_equal(model.mean.shape, mean.shape)
        assert_equal(model.concentration.shape, ())

    def test_shapes_independent_dims(self):
        samples = 10000
        mean = np.ones((3,))
        covariance = np.eye(3)
        x = np.random.multivariate_normal(mean, covariance, size=(13, samples,))
        model = VonMisesFisherTrainer().fit(x)
        assert_equal(model.mean.shape, np.tile(mean, (13, 1)).shape)
        assert_equal(model.concentration.shape, (13,))

    def test_von_mises_fisher(self):
        samples = 10000
        mean = np.ones((3,))
        mean /= np.linalg.norm(mean, axis=-1)
        concentration = 50

        # ToDo: Implement VonMisesFisher(...).sample(...)
        return

        x = VonMisesFisher(mean, concentration).sample(size=(samples,))
        model = VonMisesFisherTrainer().fit(x)
        assert_allclose(model.mean, mean, atol=0.1)
        assert_allclose(model.covariance, concentration, atol=0.1)
