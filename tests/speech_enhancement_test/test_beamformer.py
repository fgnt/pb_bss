import unittest

import numpy as np
import functools
import nt.testing as tc
from nt.speech_enhancement.beamformer import get_gev_vector, \
    _get_gev_vector, _cythonized_eig, eig
from nt.speech_enhancement.beamformer import get_lcmv_vector
from nt.speech_enhancement.beamformer import get_mvdr_vector
from nt.speech_enhancement.beamformer import get_pca_vector
from nt.speech_enhancement.beamformer import get_mvdr_vector_souden
from nt.math.vector import cos_similarity
from nt.utils.random_helper import uniform, hermitian, pos_def_hermitian


class TestBeamformerWrapper(unittest.TestCase):
    K, F, D = 2, 3, 6
    shape_psd = (F, D, D)

    @classmethod
    def setUpClass(self):
        self.shape_vector = self.shape_psd[:-1]

    def test_gev_dimensions(self):
        output = get_gev_vector(
            pos_def_hermitian(
                self.shape_psd), pos_def_hermitian(
                self.shape_psd))
        tc.assert_equal(output.shape, self.shape_vector)

    def test_mvdr_souden_dimensions(self):
        output = get_mvdr_vector_souden(
            pos_def_hermitian(
                self.shape_psd), pos_def_hermitian(
                self.shape_psd))
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


class TestCythonizedEig(unittest.TestCase):
    def test_result_equal(self):
        import time

        F = 513

        phi_XX = pos_def_hermitian(F, 6, 6)
        phi_NN = pos_def_hermitian(F, 6, 6)
        t = time.time()
        beamforming_vector = np.empty((F, 6), dtype=np.complex128)
        eigenvals = np.empty((F, 6), dtype=np.complex128)
        eigenvecs = np.empty((F, 6, 6), dtype=np.complex128)
        for f in range(F):
            eigenvals[f], eigenvecs[f] = eig(
                phi_XX[f, :, :], phi_NN[f, :, :]
            )
            beamforming_vector[f, :] = eigenvecs[f, :, np.argmax(eigenvals[f])]
        elapsed_time_python = time.time() - t

        t = time.time()
        eigenvals_c, eigenvecs_c = _cythonized_eig(phi_XX, phi_NN)
        beamforming_vector_cython = eigenvecs_c[range(F), :,
                                    np.argmax(eigenvals_c, axis=1)]
        elapsed_time_cython1 = time.time() - t

        tc.assert_allclose(
            cos_similarity(beamforming_vector, beamforming_vector_cython),
            1.0, atol=1e-6)

        # assume speedup is bigger than 5
        tc.assert_array_greater(elapsed_time_python / elapsed_time_cython1, 4)


from nt.speech_enhancement.beamformer import get_mvdr_vector_souden


class TestMvdrSouden(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        obs = np.array([[0, 0, 1], [0, 0.1, 1], [0.1, 0, 1]])
        self.PhiXX = obs.T.conj() @ obs
        np.testing.assert_allclose(self.PhiXX, [
            [0.01, 0., 0.1],
            [0., 0.01, 0.1],
            [0.1, 0.1, 3.],
        ])
        self.PhiNN = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def get_w_well_behaviour(self):
        w, = get_mvdr_vector_souden(
            self.PhiXX[None, ...],
            self.PhiNN[None, ...],
        )
        return w

    def test_well_behaviour(self):
        w = self.get_w_well_behaviour()
        assert repr(w) == 'array([0.03311258, 0.03311258, 0.99337748])', repr(w)
        w3 = get_mvdr_vector_souden([self.PhiXX] * 3, [self.PhiNN] * 3)
        np.testing.assert_allclose([w] * 3, w3)

    def test_difficulties(
            self,
    ):
        get_beamformer = get_mvdr_vector_souden

        for args in [
            (
                self.PhiXX[None, ...] * 0,
                self.PhiNN[None, ...],
            ),
            (
                self.PhiXX[None, ...],
                self.PhiNN[None, ...] * 0,
            ),
            (
                self.PhiXX[None, ...] * 0,
                self.PhiNN[None, ...] * 0,
            ),
        ]:
            w = get_beamformer(*args)
            assert repr(w) == 'array([[0., 0., 0.]])', repr(w)

        for args in [
            (
                self.PhiXX[None, ...] * np.inf,
                self.PhiNN[None, ...],
            ),
            (
                self.PhiXX[None, ...],
                self.PhiNN[None, ...] * np.inf,
            ),
            (
                self.PhiXX[None, ...] * np.inf,
                self.PhiNN[None, ...] * np.inf,
            ),
        ]:
            with tc.assert_raises(AssertionError):
                get_beamformer(*args)

    def test_difficulties_without_eps_single(self):
        def get_beamformer(A, B):
            return get_mvdr_vector_souden(
                A, B, eps=0
            )
        for args in [
            (
                self.PhiXX[None, ...] * 0,
                self.PhiNN[None, ...],
            ),
            (
                self.PhiXX[None, ...],
                self.PhiNN[None, ...] * 0,
            ),
            (
                self.PhiXX[None, ...] * 0,
                self.PhiNN[None, ...] * 0,
            ),
            (
                self.PhiXX[None, ...] * np.inf,
                self.PhiNN[None, ...],
            ),
            (
                self.PhiXX[None, ...],
                self.PhiNN[None, ...] * np.inf,
            ),
            (
                self.PhiXX[None, ...] * np.inf,
                self.PhiNN[None, ...] * np.inf,
            ),
        ]:
            with tc.assert_raises(AssertionError):
                get_beamformer(*args)

    def test_difficulties_eps_multi(self):
        """
        Broken frequencies only damage total result, when they have inf
        values. Zeros are ok for target_psd_matrix and noise_psd_matrix.
        """
        well_w = self.get_w_well_behaviour()

        def get_beamformer(A, B):
            return get_mvdr_vector_souden(
                A, B,
                return_ref_channel=True
            )

        for args in [
            (
                [self.PhiXX * 0, self.PhiXX],
                [self.PhiNN, self.PhiNN],
            ),
            (
                [self.PhiXX, self.PhiXX],
                [self.PhiNN * 0, self.PhiNN],
            ),
            (
                [self.PhiXX * 0, self.PhiXX],
                [self.PhiNN * 0, self.PhiNN],
            ),
        ]:
            w, ref_channel = get_beamformer(*args)
            assert ref_channel == 2, ref_channel
            np.testing.assert_allclose(
                w,
                np.array([[0., 0., 0.], well_w])
            )

        for args in [
            (
                [self.PhiXX * np.inf, self.PhiXX],
                [self.PhiNN, self.PhiNN],
            ),
            (
                [self.PhiXX, self.PhiXX],
                [self.PhiNN * np.inf, self.PhiNN],
            ),
            (
                [self.PhiXX * np.inf, self.PhiXX],
                [self.PhiNN * np.inf, self.PhiNN],
            ),
        ]:
            with tc.assert_raises(AssertionError):
                get_beamformer(*args)

    def test_difficulties_without_eps_multi(self):
        """
        When eps is set to zero, any broken frequency will damage the
        ref_channel estimation. Independent if target_psd_matrix or
        noise_psd_matrix is zero or inf.
        """

        def get_beamformer(A, B):
            return get_mvdr_vector_souden(
                A, B,
                eps=0,
                return_ref_channel=True
            )

        for args in [
            (
                [self.PhiXX * 0, self.PhiXX],
                [self.PhiNN, self.PhiNN],
            ),
            (
                [self.PhiXX, self.PhiXX],
                [self.PhiNN * 0, self.PhiNN],
            ),
            (
                [self.PhiXX * 0, self.PhiXX],
                [self.PhiNN * 0, self.PhiNN],
            ),
            (
                [self.PhiXX * np.inf, self.PhiXX],
                [self.PhiNN, self.PhiNN],
            ),
            (
                [self.PhiXX, self.PhiXX],
                [self.PhiNN * np.inf, self.PhiNN],
            ),
            (
                [self.PhiXX * np.inf, self.PhiXX],
                [self.PhiNN * np.inf, self.PhiNN],
            ),
        ]:
            with tc.assert_raises(AssertionError):
                get_beamformer(*args)
