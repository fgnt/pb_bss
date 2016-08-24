import unittest
import nt.testing as tc
from nose_parameterized import parameterized, param
import numpy as np
from nt.speech_enhancement.mask_estimation import simple_ideal_soft_mask
from nt.speech_enhancement.mask_module import ideal_binary_mask
from nt.speech_enhancement.mask_module import wiener_like_mask
from nt.utils.random_helper import randn
import nt.testing as tc

from nt.speech_enhancement import mask_module


def randc128(*shape):
    return randn(*shape, dtype=np.complex128)


class IdealBinaryMaskTest(unittest.TestCase):

    params = [
        param('default 2D',
              kwargs={'signal': randc128(2, 3)},
              out_shape=(2, 3),
              out_source_axis=0,),
    ]+[
        param('sweep source_axis 2D {}'.format(source_axis),
              kwargs={'signal': randc128(2, 3), 'source_axis': source_axis},
              out_shape=(2, 3),
              out_source_axis=source_axis, )
        for source_axis in range(2)
    ]+[
        param('sweep sensor_axis 2D {} {}'.format(source_axis, sensor_axis),
              kwargs={'signal': randc128(2, 3),
                      'source_axis': source_axis,
                      'sensor_axis': sensor_axis},
              out_shape=out_shape,
              out_source_axis=0, )
        for out_shape, source_axis in zip([(2,), (3,)], range(2))
        for sensor_axis in range(2) if source_axis != sensor_axis
    ]+[
        param('sweep source_axis 3D',
              kwargs={'signal': randc128(2, 3, 4), 'source_axis': source_axis},
              out_shape=(2, 3, 4),
              out_source_axis=source_axis, )
        for source_axis in range(3)
    ]+[
        param('sweep sensor_axis 3D {}, {}'.format(source_axis, sensor_axis),
              kwargs={'signal': randc128(2, 3, 4),
                      'source_axis': source_axis,
                      'sensor_axis': sensor_axis},
              out_shape=out_shape,
              out_source_axis=source_axis if sensor_axis > source_axis
                                          else source_axis - 1, )
        for source_axis in range(3)
        for out_shape, sensor_axis in zip([(3, 4), (2, 4), (2, 3)], range(3))
        if source_axis != sensor_axis
    ]

    pass_binary_result = False
    pass_greater_equal_zero = False
    pass_smaler_equal_one = False
    pass_sum_to_one = False
    sensor_axis_allowed = True

    def setUp(self):
        self.mask_calculator = mask_module.ideal_binary_mask

    @parameterized.expand(params)
    def test_shape(self, name, kwargs, out_shape, out_source_axis):
        if not self.sensor_axis_allowed:
            if len(kwargs) > 2:
                with tc.assert_raises(AssertionError):
                    self.mask_calculator(**kwargs)
                return
        out = self.mask_calculator(**kwargs)
        np.testing.assert_array_equal(out.shape, out_shape)

    @parameterized.expand(params)
    def test_binary_result(self, name, kwargs, out_shape, out_source_axis):
        if self.pass_binary_result == True:
            return
        if not self.sensor_axis_allowed:
            if len(kwargs) > 2:
                with tc.assert_raises(AssertionError):
                    self.mask_calculator(**kwargs)
                return
        print(self.pass_binary_result)
        out = self.mask_calculator(**kwargs)
        np.testing.assert_array_equal(np.unique(out), [0, 1])

    @parameterized.expand(params)
    def test_greater_equal_zero(self, name, kwargs, out_shape, out_source_axis):
        if self.pass_greater_equal_zero:
            return
        if not self.sensor_axis_allowed:
            if len(kwargs) > 2:
                with tc.assert_raises(AssertionError):
                    self.mask_calculator(**kwargs)
                return
        out = self.mask_calculator(**kwargs)
        tc.assert_array_greater_equal(np.min(out), 0)

    @parameterized.expand(params)
    def test_smaler_equal_one(self, name, kwargs, out_shape, out_source_axis):
        if self.pass_smaler_equal_one:
            return
        if not self.sensor_axis_allowed:
            if len(kwargs) > 2:
                with tc.assert_raises(AssertionError):
                    self.mask_calculator(**kwargs)
                return
        out = self.mask_calculator(**kwargs)
        tc.assert_array_less_equal(np.max(out), 1)

    @parameterized.expand(params)
    def test_sum_to_one(self, name, kwargs, out_shape, out_source_axis):
        if self.pass_sum_to_one:
            return
        if not self.sensor_axis_allowed:
            if len(kwargs) > 2:
                with tc.assert_raises(AssertionError):
                    self.mask_calculator(**kwargs)
                return
        out = self.mask_calculator(**kwargs)
        np.testing.assert_allclose(np.sum(out, out_source_axis), 1)


class WienerLikeMaskTest(IdealBinaryMaskTest):
    def setUp(self):
        self.mask_calculator = mask_module.wiener_like_mask
        self.pass_binary_result = True


class IdealAmplitudeMaskTest(IdealBinaryMaskTest):

    pass_binary_result = True
    pass_smaler_equal_one = True
    pass_sum_to_one = True
    sensor_axis_allowed = False

    def setUp(self):
        self.mask_calculator = mask_module.ideal_amplitude_mask


class IdealRatioMaskTest(IdealBinaryMaskTest):

    pass_binary_result = True
    sensor_axis_allowed = False

    def setUp(self):
        self.mask_calculator = mask_module.ideal_ratio_mask


class IdealComplexMaskTest(IdealBinaryMaskTest):

    pass_binary_result = True
    pass_smaler_equal_one = True
    pass_greater_equal_zero = True
    # pass_sum_to_one = True
    sensor_axis_allowed = False

    def setUp(self):
        self.mask_calculator = mask_module.ideal_complex_mask


class PhaseSensitiveMaskTest(IdealBinaryMaskTest):

    pass_binary_result = True
    pass_smaler_equal_one = True
    pass_greater_equal_zero = True
    # pass_sum_to_one = True
    sensor_axis_allowed = False

    def setUp(self):
        self.mask_calculator = mask_module.phase_sensitive_mask


F, T, D, K = 51, 31, 6, 2
X_all = randc128(F, T, D, K)
X, N = (X_all[:, :, :, 0], X_all[:, :, :, 1])


class TestIdealBinaryMask(unittest.TestCase):
    def setUp(self):
        self.signal = randc128(K, D, F, T)

    def test_without_parameters(self):
        mask = ideal_binary_mask(self.signal)
        tc.assert_equal(mask.shape, (K, D, F, T))

    def test_feature_axis(self):
        mask = ideal_binary_mask(self.signal, sensor_axis=1)
        tc.assert_equal(mask.shape, (K, F, T))

    def test_component_and_feature_axis(self):
        mask = ideal_binary_mask(self.signal, source_axis=2, sensor_axis=3)
        tc.assert_equal(mask.shape, (K, D, F))

    def test_equal_power(self):
        signal = np.asarray([0.5 + 0.5j, 0.5 + 0.5j])
        mask = ideal_binary_mask(signal)
        tc.assert_equal(mask, [1, 0])


class TestWienerLikeMask(unittest.TestCase):
    def setUp(self):
        self.signal = randc128(K, D, F, T)

    def test_without_parameters(self):
        mask = wiener_like_mask(self.signal)
        tc.assert_equal(mask.shape, (K, D, F, T))

    def test_feature_axis(self):
        mask = wiener_like_mask(self.signal, sensor_axis=1)
        tc.assert_equal(mask.shape, (K, F, T))

    def test_component_and_feature_axis(self):
        mask = wiener_like_mask(self.signal, source_axis=2, sensor_axis=3)
        tc.assert_equal(mask.shape, (K, D, F))

    def test_equal_power(self):
        signal = np.asarray([0.5 + 0.5j, 0.5 + 0.5j])
        mask = wiener_like_mask(signal)
        tc.assert_equal(mask, [0.5, 0.5])

if __name__ == '__main__':

    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
