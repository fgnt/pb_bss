import unittest
from nt.utils.random import uniform, hermitian
import nt.testing as tc
from nt.utils.math_ops import cos_similarity
import numpy as np

from nt.speech_enhancement.beamformer import get_gev_vector
from nt.speech_enhancement.beamformer import get_lcmv_vector
from nt.speech_enhancement.beamformer import get_mvdr_vector
from nt.speech_enhancement.beamformer import get_pca_vector


class TestBeamformerWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        K, F, D = 2, 3, 6
        self.shapes = [
            ((D, D), (D,)),
            ((F, D, D), (F, D)),
            ((K, F, D, D), (K, F, D)),
        ]

    def test_gev_dimensions(self):
        for shape in self.shapes:
            output = get_gev_vector(uniform(shape[0]), uniform(shape[0]))
            assert output.shape == shape[1]

    def test_pca_dimensions(self):
        for shape in self.shapes:
            output = get_pca_vector(uniform(shape[0]))
            assert output.shape == shape[1]

    def test_mvdr_dimensions(self):
        for shape in self.shapes:
            output = get_mvdr_vector(uniform(shape[1]), uniform(shape[0]))
            assert output.shape == shape[1]

    def test_lcmv_dimensions(self):
        K, F, D = 2, 3, 6
        output = get_lcmv_vector(uniform((K, F, D)), [1, 0], uniform((F, D, D)))
        assert output.shape == (F, D)

    def test_gev_falls_back_to_pca_for_unity_noise_matrix(self):
        Phi_XX = hermitian(6, 6)
        Phi_NN = np.identity(6)
        W_gev = get_gev_vector(Phi_XX, Phi_NN)
        W_pca = get_pca_vector(Phi_XX)

        tc.assert_equal(cos_similarity(W_gev, W_pca), 1.0)
