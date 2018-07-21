import numpy as np
from numpy.testing import assert_allclose
import unittest
from dc_integration.distribution import complex_angular_central_gaussian

Trainer = complex_angular_central_gaussian.ComplexAngularCentralGaussianTrainer


class TestComplexAngularCentralGaussian(unittest.TestCase):
    def test_complex_angular_central_gaussian_identity_matrix(self):
        samples = 10000
        mean = np.zeros((3,))
        covariance = np.eye(3)
        covariance /= np.trace(covariance)
        x = np.random.multivariate_normal(
            mean, covariance, size=(samples,)
        ) + 1j * np.random.multivariate_normal(
            mean, covariance, size=(samples,)
        )
        x /= np.linalg.norm(x, axis=-1, keepdims=True)
        model = Trainer().fit(x)
        assert_allclose(model.covariance, covariance, atol=0.1)

    def test_complex_angular_central_gaussian(self):
        samples = 10000
        mean = np.zeros((3,))
        N, D = 10000, 3
        covariance = np.array([
            [10, 1+1j, 1+1j],
            [1-1j, 5, 1],
            [1-1j, 1, 2]
        ])
        covariance = covariance + covariance.conj().T
        covariance /= np.trace(covariance)
        x = 1 / np.sqrt(2) * np.random.normal(size=(N, D)) + 1j / np.sqrt(2) * np.random.normal(size=(N, D))
        cholesky = np.linalg.cholesky(covariance)
        x = (cholesky @ x.T).T
        x /= np.linalg.norm(x, axis=-1, keepdims=True)
        sanity = np.einsum('nd,nD->dD', x, x.conj())
        sanity /= np.trace(sanity)
        print(sanity)
        model = Trainer().fit(x, hermitize=False)
        print(model.covariance)
        assert_allclose(model.covariance, covariance, atol=0.1)
