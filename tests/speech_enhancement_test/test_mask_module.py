import unittest
import nt.testing as tc
import numpy as np
from nt.speech_enhancement.mask_estimation import simple_ideal_soft_mask
from nt.speech_enhancement.mask_module import ideal_binary_mask
from nt.speech_enhancement.mask_module import wiener_like_mask


def _random_stft(*shape):
    return np.random.rand(*shape) + 1j * np.random.rand(*shape)


F, T, D, K = 51, 31, 6, 2
X_all = _random_stft(F, T, D, K)
X, N = (X_all[:, :, :, 0], X_all[:, :, :, 1])


class SimpleIdealSoftMaskTests(unittest.TestCase):
    def test_single_input(self):
        M1 = simple_ideal_soft_mask(X_all)
        tc.assert_equal(M1.shape, (51, 31, 2))
        tc.assert_almost_equal(np.sum(M1, axis=2), 1)
        return M1

    def test_separate_input(self):
        M2 = simple_ideal_soft_mask(X, N)
        tc.assert_equal(M2.shape, (51, 31, 2))
        tc.assert_almost_equal(np.sum(M2, axis=2), 1)
        return M2

    def test_separate_input_equals_single_input(self):
        tc.assert_equal(self.test_single_input(), self.test_separate_input())

    def test_(self):
        M3 = simple_ideal_soft_mask(X_all, N)
        tc.assert_equal(M3.shape, (51, 31, 3))
        tc.assert_almost_equal(np.sum(M3, axis=2), 1)

    def test_negative_feature_bin(self):
        M4 = simple_ideal_soft_mask(X, N, feature_dim=-3)
        tc.assert_equal(M4.shape, (51, 6, 2))
        tc.assert_almost_equal(np.sum(M4, axis=2), 1)


class TestIdealBinaryMask(unittest.TestCase):
    def setUp(self):
        self.signal = _random_stft(K, D, F, T)

    def test_without_parameters(self):
        mask = ideal_binary_mask(self.signal)
        tc.assert_equal(mask.shape, (K, D, F, T))

    def test_feature_axis(self):
        mask = ideal_binary_mask(self.signal, feature_axis=1)
        tc.assert_equal(mask.shape, (K, F, T))

    def test_component_and_feature_axis(self):
        mask = ideal_binary_mask(self.signal, component_axis=2, feature_axis=3)
        tc.assert_equal(mask.shape, (K, D, F))

    def test_forbidden_list_input(self):
        with self.assertRaises(AttributeError):
            ideal_binary_mask([self.signal, self.signal])

    def test_equal_power(self):
        signal = np.asarray([0.5 + 0.5j, 0.5 + 0.5j])
        mask = ideal_binary_mask(signal)
        tc.assert_equal(mask, [1, 0])


class TestWienerLikeMask(unittest.TestCase):
    def setUp(self):
        self.signal = _random_stft(K, D, F, T)

    def test_without_parameters(self):
        mask = wiener_like_mask(self.signal)
        tc.assert_equal(mask.shape, (K, D, F, T))

    def test_feature_axis(self):
        mask = wiener_like_mask(self.signal, feature_axis=1)
        tc.assert_equal(mask.shape, (K, F, T))

    def test_component_and_feature_axis(self):
        mask = wiener_like_mask(self.signal, component_axis=2, feature_axis=3)
        tc.assert_equal(mask.shape, (K, D, F))

    def test_forbidden_list_input(self):
        with self.assertRaises(AttributeError):
            wiener_like_mask([self.signal, self.signal])

    def test_equal_power(self):
        signal = np.asarray([0.5 + 0.5j, 0.5 + 0.5j])
        mask = wiener_like_mask(signal)
        tc.assert_equal(mask, [0.5, 0.5])
