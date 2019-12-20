import unittest

import numpy as np

import numpy.testing as tc
from pb_bss.extraction import get_power_spectral_density_matrix
from pb_bss.testing.module_asserts import (
    assert_hermitian,
    assert_positive_semidefinite,
)
from pb_bss.extraction.beamformer import get_power_spectral_density_matrix


def rand(*shape, data_type):
    if not shape:
        shape = (1,)
    elif isinstance(shape[0], tuple):
        shape = shape[0]

    def uniform(data_type_local):
        return np.random.uniform(-1, 1, shape).astype(data_type_local)

    if data_type in (np.float32, np.float64):
        return uniform(data_type)
    elif data_type is np.complex64:
        return uniform(np.float32) + 1j * uniform(np.float32)
    elif data_type is np.complex128:
        return uniform(np.float64) + 1j * uniform(np.float64)


class TestPowerSpectralDensityMatrix(unittest.TestCase):
    # or covariance matrix

    F, T, D, K = 51, 31, 6, 2

    def generate_date(self, x_shape, mask_shape):
        x = rand(x_shape, data_type=np.complex128)
        mask = np.random.uniform(0, 1, mask_shape)
        mask = mask / np.sum(mask, axis=0, keepdims=True)
        return x, mask

    def generate_and_verify_psd(self, x_shape, mask_shape, psd_shape=None):
        x, mask = self.generate_date(x_shape, mask_shape)
        if mask_shape is None:
            psd = get_power_spectral_density_matrix(x)
        else:
            psd = get_power_spectral_density_matrix(x, mask)
        if psd_shape is not None:
            tc.assert_equal(psd.shape, psd_shape)
        assert_hermitian(psd)
        assert_positive_semidefinite(psd)

    def test_PSD_without_mask(self):
        self.gererate_and_verify_psd((self.D, self.T), None, psd_shape=(self.D, self.D))

    def test_PSD_with_mask(self):
        self.gererate_and_verify_psd((self.D, self.T), (self.T,), psd_shape=(self.D, self.D))

    def test_PSD_with_mask_with_source(self):
        self.gererate_and_verify_psd((self.D, self.T), (self.K, self.T,), psd_shape=(self.K, self.D, self.D))

    def test_PSD_with_mask_independent_dim(self):
        self.gererate_and_verify_psd((self.F, self.D, self.T), (self.F, self.T,), psd_shape=(self.F, self.D, self.D))
        self.gererate_and_verify_psd((self.F, self.F, self.D, self.T), (self.F, self.F, self.T,),
                                     psd_shape=(self.F, self.F, self.D, self.D))

    def test_PSD_with_mask_independent_dim_with_source(self):
        self.gererate_and_verify_psd((self.F, self.D, self.T), (self.F, self.K, self.T,),
                                     psd_shape=(self.F, self.K, self.D, self.D))
        self.gererate_and_verify_psd((self.F, self.F, self.D, self.T), (self.F, self.F, self.K, self.T,),
                                     psd_shape=(self.F, self.F, self.K, self.D, self.D))

    def test_PSD_without_mask_independent_dim(self):
        self.gererate_and_verify_psd((self.F, self.D, self.T), None, psd_shape=(self.F, self.D, self.D))
        self.gererate_and_verify_psd((self.F, self.F, self.D, self.T), (self.F, self.F, self.T,),
                                     psd_shape=(self.F, self.F, self.D, self.D))

    def test_predict_output(self):
        x, _ = self.generate_date((self.D,), None)
        x_rep = x[:, np.newaxis].repeat((self.T,), 1)
        psd = get_power_spectral_density_matrix(x_rep)

        psd_predict = x[:, np.newaxis].dot(x[np.newaxis, :].conj())
        tc.assert_almost_equal(psd, psd_predict)

    def test_predict_output_with_mask(self):
        x, _ = self.generate_date((self.D,), None)
        x_rep = x[:, np.newaxis].repeat((self.T,), 1)
        mask = np.ones((self.T,))
        psd = get_power_spectral_density_matrix(x_rep, mask)
        psd2 = get_power_spectral_density_matrix(x_rep, mask * 2)
        psd3 = get_power_spectral_density_matrix(x_rep, mask * 0.5)

        psd_predict = x[:, np.newaxis].dot(x[np.newaxis, :].conj())
        tc.assert_almost_equal(psd, psd_predict)
        tc.assert_almost_equal(psd, psd2)
        tc.assert_almost_equal(psd, psd3)

    def test_different_valued_masks_output(self):
        x, mask = self.generate_date((self.F, self.D, self.T), (self.F, self.T,))

        psd = get_power_spectral_density_matrix(x, mask)
        psd2 = get_power_spectral_density_matrix(x, mask*2)
        psd3 = get_power_spectral_density_matrix(x, mask*0.5)

        tc.assert_almost_equal(psd, psd2)
        tc.assert_almost_equal(psd, psd3)


class TestCovariance(unittest.TestCase):
    def generate_data(self, x_shape, mask_shape):
        x = rand(x_shape, data_type=np.complex128)
        mask = np.random.uniform(0, 1, mask_shape)
        mask = mask / np.sum(mask, axis=0, keepdims=True)
        return x, mask

    def test_covariance_without_mask(self):
        x = rand(3, 4)
        psd = get_power_spectral_density_matrix(x)
        tc.assert_equal(psd.shape, (3, 3))
        tc.assert_positive_semidefinite(psd)

    def test_covariance_with_mask(self):
        x = rand(3, 4)
        mask = np.random.uniform(0, 1, (4,))
        psd = get_power_spectral_density_matrix(x, mask)
        tc.assert_equal(psd.shape, (3, 3))
        tc.assert_positive_semidefinite(psd)

    def test_covariance_with_mask_with_source(self):
        x = rand(3, 4)
        mask = np.random.uniform(0, 1, (2, 4))
        psd = get_power_spectral_density_matrix(x[None, ...], mask)
        tc.assert_equal(psd.shape, (2, 3, 3))
        tc.assert_positive_semidefinite(psd)

    def test_covariance_with_mask_independent_dim(self):
        x = rand(2, 3, 4)
        mask = np.random.uniform(0, 1, (2, 4,))
        psd = get_power_spectral_density_matrix(x, mask)
        tc.assert_equal(psd.shape, (2, 3, 3))
        tc.assert_positive_semidefinite(psd)

    def test_covariance_without_mask_independent_dim(self):
        x = rand(1, 2, 3, 4)
        psd = get_power_spectral_density_matrix(x)
        tc.assert_equal(psd.shape, (1, 2, 3, 3))
        tc.assert_positive_semidefinite(psd)

    def test_multiple_sources_for_source_separation(self):
        x = rand(2, 3, 4)
        mask = np.random.uniform(0, 1, (5, 2, 4,))
        psd = get_power_spectral_density_matrix(x[np.newaxis, ...], mask)
        tc.assert_equal(psd.shape, (5, 2, 3, 3))
        tc.assert_positive_semidefinite(psd)
