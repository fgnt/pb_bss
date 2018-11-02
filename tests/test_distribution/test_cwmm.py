import numpy as np
from numpy.testing import assert_equal
import unittest
from dc_integration.distribution import CWMMTrainer
from dc_integration.distribution import ComplexAngularCentralGaussian


class TestCWMM(unittest.TestCase):
    def test_cwmm_shape(self):
        # TODO: Sample from the correct distribution please
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

        model = CWMMTrainer().fit(x, num_classes=2)

        assert_equal(model.weight.shape, (2,))
        assert_equal(model.complex_watson.mode.shape, (2, 3))
        assert_equal(model.complex_watson.concentration.shape, (2,))
