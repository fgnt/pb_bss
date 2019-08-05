import unittest
import numpy as np
from paderbox.speech_enhancement.beamformer import get_bf_vector
from paderbox.speech_enhancement.beamformer import zero_degree_normalization
from paderbox.utils.random_utils import pos_def_hermitian


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


class TestBeamformerWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        sensors, bins, num_spk = 5, 128, 2
        self.num_spk = num_spk
        self.target_psd = pos_def_hermitian(bins, sensors, sensors)
        self.noise_psd = pos_def_hermitian(bins, sensors, sensors)
        self.output_shape = (bins, sensors)

    def pca_shape(self):
        output = get_bf_vector(
            f'pca', self.target_psd, self.noise_psd)
        assert output.shape == self.output_shape

    def test_mvdr_shape(self):
        bf = 'mvdr'
        for atf in ['pca', 'scaled_gev_atf']:
            output = get_bf_vector(
                f'{atf}+{bf}', self.target_psd, self.noise_psd)
            assert output.shape == self.output_shape

    def test_mvdr_souden_shape(self):
        bf = 'mvdr_souden'
        for rank1 in ['rank1_pca', 'rank1_gev']:
            output = get_bf_vector(
                f'{rank1}+{bf}', self.target_psd, self.noise_psd)
            assert output.shape == self.output_shape

    def test_wmwf_shape(self):
        bf = 'wmwf'
        for rank1 in ['rank1_pca', 'rank1_gev']:
            output = get_bf_vector(
                f'{rank1}+{bf}', self.target_psd, self.noise_psd)
            assert output.shape == self.output_shape

    def test_rank1_gev_gev(self):
        gev_rank1 = get_bf_vector(
            'rank1_gev+gev', self.target_psd, self.noise_psd)
        assert gev_rank1.shape == self.output_shape, gev_rank1.shape
        gev_rank1_test_equal = get_bf_vector(
            'rank1_gev+gev', self.target_psd, self.noise_psd)
        np.testing.assert_equal(gev_rank1, gev_rank1_test_equal)
        gev = get_bf_vector('gev', self.target_psd, self.noise_psd)
        np.testing.assert_allclose(
            np.abs(gev_rank1), np.abs(gev), verbose=True)
        np.testing.assert_allclose(zero_degree_normalization(gev_rank1, 0),
                                   zero_degree_normalization(gev, 0),
                                   verbose=True)
