import numpy as np
from numpy.testing import assert_equal
import unittest
from pb_bss.distribution import (
    ComplexAngularCentralGaussian,
    ComplexWatsonTrainer,
)


class TestComplexWatson(unittest.TestCase):
    def test_complex_watson_shapes(self):
        # TODO: Sample from the correct distribution please
        covariance = np.array(
            [[10, 1 + 1j, 1 + 1j], [1 - 1j, 5, 1], [1 - 1j, 1, 2]]
        )
        covariance /= np.trace(covariance)
        model = ComplexAngularCentralGaussian.from_covariance(
            covariance=covariance,
        )
        x = model.sample(size=(10000,))
        model = ComplexWatsonTrainer().fit(x)
        assert_equal(model.mode.shape, (3,))
        assert_equal(model.concentration.shape, ())
