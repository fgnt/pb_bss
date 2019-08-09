import numpy as np
import unittest
from pb_bss.distribution import (
    ComplexAngularCentralGaussian,
    ComplexAngularCentralGaussianTrainer,
)


class TestComplexAngularCentralGaussian(unittest.TestCase):
    def test_complex_angular_central_gaussian(self):
        atol = 0.01

        covariance = np.array(
            [[10, 1 + 1j, 1 + 1j], [1 - 1j, 5, 1], [1 - 1j, 1, 2]]
        )
        covariance /= np.trace(covariance)
        model = ComplexAngularCentralGaussian.from_covariance(
            covariance=covariance
        )
        x = model.sample(size=(10000,))
        model = ComplexAngularCentralGaussianTrainer().fit(
            x,
            covariance_norm='trace'
        )
        np.testing.assert_allclose(model.covariance, covariance, atol=atol)

        model = ComplexAngularCentralGaussianTrainer().fit(x)
        with np.testing.assert_raises(AssertionError):
            # Scaling of covariance is eigenvalue
            np.testing.assert_allclose(model.covariance, covariance, atol=atol)
        model_covariance = model.covariance / np.trace(model.covariance)
        np.testing.assert_allclose(model_covariance, covariance, atol=atol)
