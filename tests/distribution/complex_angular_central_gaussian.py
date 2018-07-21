import numpy as np
from numpy.testing import assert_allclose
import unittest
from dc_integration.distribution import (
    ComplexAngularCentralGaussian,
    ComplexAngularCentralGaussianTrainer,
)


class TestComplexAngularCentralGaussian(unittest.TestCase):
    def test_complex_angular_central_gaussian(self):
        covariance = np.array(
            [[10, 1 + 1j, 1 + 1j], [1 - 1j, 5, 1], [1 - 1j, 1, 2]]
        )
        covariance /= np.trace(covariance)
        model = ComplexAngularCentralGaussian(covariance=covariance)
        x = model.sample(size=(10000,))
        model = ComplexAngularCentralGaussianTrainer().fit(x)
        assert_allclose(model.covariance, covariance, atol=0.1)
