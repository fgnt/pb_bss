import unittest

import numpy as np
from parameterized import parameterized, param

import numpy.testing as tc
from pb_bss.extraction import mask_module
from pb_bss.extraction.mask_module import wiener_like_mask, lorenz_mask, \
    ideal_binary_mask
from pb_bss.testing.random_utils import randn
from pb_bss.testing.module_asserts import (
    assert_array_less_equal,
    assert_array_greater_equal,
)


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
    pass_smaller_equal_one = False
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
        assert_array_greater_equal(np.min(out), 0)

    @parameterized.expand(params)
    def test_smaler_equal_one(self, name, kwargs, out_shape, out_source_axis):
        if self.pass_smaller_equal_one:
            return
        if not self.sensor_axis_allowed:
            if len(kwargs) > 2:
                with tc.assert_raises(AssertionError):
                    self.mask_calculator(**kwargs)
                return
        out = self.mask_calculator(**kwargs)
        assert_array_less_equal(np.max(out), 1)

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
    pass_smaller_equal_one = True
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
    pass_smaller_equal_one = True
    pass_greater_equal_zero = True
    # pass_sum_to_one = True
    sensor_axis_allowed = False

    def setUp(self):
        self.mask_calculator = mask_module.ideal_complex_mask


class PhaseSensitiveMaskTest(IdealBinaryMaskTest):

    pass_binary_result = True
    pass_smaller_equal_one = True
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


class TestLorenzMask(unittest.TestCase):
    def setUp(self):
        self.signal = randc128(K, D, F, T)

    params_sensor_axis = [
        param('sensor_axis_None', sensor_axis=None),
        param('sensor_axis_0', sensor_axis=0),
        param('sensor_axis_1', sensor_axis=1),
    ]

    params_lorenz_fraction = [
        param('lorenz_fraction_{}'.format(lorenz_fraction),
              lorenz_fraction=lorenz_fraction)
        for lorenz_fraction in (0.1, 0.4, 0.8, 0.89)
        ]

    params_weight = [
        param('weight_{}'.format(weight), weight=weight)
        for weight in (0, 0.5, 0.999)
        ]

    @parameterized.expand(
        params_sensor_axis + params_lorenz_fraction + params_weight)
    def test_shape(self, name, sensor_axis=None, lorenz_fraction=0.98,
                   weight=0.999):
        mask = lorenz_mask(
            self.signal,
            sensor_axis=sensor_axis,
            lorenz_fraction=lorenz_fraction,
            weight=weight
        )
        tc.assert_equal(
            mask.shape,
            tuple(v for i, v in enumerate([K, D, F, T]) if i != sensor_axis)
        )

    @parameterized.expand(params_sensor_axis + params_lorenz_fraction)
    def test_weight_0(self, name, sensor_axis=None, lorenz_fraction=0.98):
        mask = lorenz_mask(
            self.signal,
            sensor_axis=sensor_axis,
            lorenz_fraction=lorenz_fraction,
            weight=0
        )
        tc.assert_equal(mask, np.ones_like(mask) / 2)

    @parameterized.expand(
        params_sensor_axis + params_lorenz_fraction + params_weight)
    def test_in_bound(self, name, sensor_axis=None, lorenz_fraction=0.98,
                      weight=0.999):
        mask = lorenz_mask(
            self.signal,
            sensor_axis=sensor_axis,
            lorenz_fraction=lorenz_fraction,
            weight=weight
        )
        assert_array_greater_equal(mask, 0.5 * (1 - weight))
        assert_array_less_equal(mask, 0.5 * (1 + weight))

    def test_multi_channel(self):
        signal = self.signal[0][0]
        mask1 = lorenz_mask(signal)
        mask2 = lorenz_mask([signal, signal])
        tc.assert_equal(mask1, mask2[0])
        tc.assert_equal(mask1, mask2[1])

    def test_mask_single_channel(self):
        shape = (3, 3)
        signal = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)
        mask = lorenz_mask(signal, weight=1)

        tc.assert_equal(mask, np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 1],
                                         dtype=np.float32).reshape(shape))

    def test_mask_two_channels(self):
        shape = (2, 3, 3)
        signal = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)
        mask = lorenz_mask(signal, weight=1)

        tc.assert_equal(mask, np.asarray([
            [[0, 0, 0],
             [0, 1, 1],
             [1, 1, 1]],
            [[0, 0, 1],
             [1, 1, 1],
             [1, 1, 1]]], dtype=np.float32))


if __name__ == '__main__':

    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
