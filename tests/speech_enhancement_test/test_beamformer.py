import unittest

import numpy as np

import nt.testing as tc
from nt.speech_enhancement.beamformer import get_gev_vector, _get_gev_vector
from nt.speech_enhancement.beamformer import get_lcmv_vector
from nt.speech_enhancement.beamformer import get_mvdr_vector
from nt.speech_enhancement.beamformer import get_pca_vector
from nt.utils.math_ops import cos_similarity
from nt.utils.random import uniform, hermitian, pos_def_hermitian


class TestBeamformerWrapper(unittest.TestCase):
    K, F, D = 2, 3, 6
    shape_psd = (F, D, D)

    @classmethod
    def setUpClass(self):
        self.shape_vector = self.shape_psd[:-1]

    def test_gev_dimensions(self):
        output = get_gev_vector(pos_def_hermitian(self.shape_psd), pos_def_hermitian(self.shape_psd))
        tc.assert_equal(output.shape, self.shape_vector)

    def test_pca_dimensions(self):
        output = get_pca_vector(uniform(self.shape_psd))
        assert output.shape == self.shape_vector

    def test_mvdr_dimensions(self):
        output = get_mvdr_vector(uniform(self.shape_vector), uniform(self.shape_psd))
        assert output.shape == self.shape_vector

    def test_lcmv_dimensions(self):
        K, F, D = self.K, self.F, self.D
        output = get_lcmv_vector(uniform((K, F, D)), [1, 0], uniform((F, D, D)))
        assert output.shape == (F, D)

    def test_gev_falls_back_to_pca_for_unity_noise_matrix(self):
        Phi_XX = hermitian(6, 6)
        Phi_NN = np.identity(6)
        W_gev = get_gev_vector(Phi_XX, Phi_NN)
        W_pca = get_pca_vector(Phi_XX)

        tc.assert_allclose(cos_similarity(W_gev, W_pca), 1.0, atol=1e-6)


class TestBeamformerWrapperWithoutIndependent(TestBeamformerWrapper):
    K, F, D = 2, 3, 6
    shape_psd = (D, D)


class TestBeamformerWrapperWithSpeakers(TestBeamformerWrapper):
    K, F, D = 2, 3, 6
    shape_psd = (K, F, D, D)


class TestCythonizedGetGEV(unittest.TestCase):
    def test_import(self):
        from nt.speech_enhancement.cythonized.get_gev_vector import \
            _c_get_gev_vector

    def test_result_equal(self):
        import time

        F = 513

        phi_XX = pos_def_hermitian(F, 6, 6)
        phi_NN = pos_def_hermitian(F, 6, 6)
        t = time.time()
        python_gev = _get_gev_vector(phi_XX, phi_NN)
        elapsed_time_python = time.time() - t

        t = time.time()
        cython_gev = get_gev_vector(phi_XX, phi_NN, True)
        elapsed_time_cython1 = time.time() - t

        tc.assert_allclose(cos_similarity(python_gev, cython_gev),
                           1.0, atol=1e-6)

        # assume speedup is bigger than 5
        tc.assert_array_greater(elapsed_time_python/elapsed_time_cython1, 5)
