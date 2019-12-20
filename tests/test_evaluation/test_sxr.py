import numpy as np
import numpy.testing as nptest
import unittest

from pb_bss.evaluation.sxr_module import input_sxr, output_sxr
from paderbox.array import morph
from paderbox.testing import condition


# ToDo: move this test to pb_bss


class TestSXR(unittest.TestCase):
    def setUp(self):
        samples = 8000
        self.s1 = np.random.normal(size=(samples,))
        self.s2 = np.random.normal(size=(samples,))
        self.n = np.random.normal(size=(samples,))

        self.s1 /= np.sqrt(np.mean(self.s1 ** 2))
        self.s2 /= np.sqrt(np.mean(self.s2 ** 2))
        self.n /= np.sqrt(np.mean(self.n ** 2))

    def test_input_sxr(self):
        images = morph('kt->k1t', 10 * np.stack((self.s1, self.s2)))
        noise = morph('t->1t', self.n)
        sdr, sir, snr = input_sxr(images, noise, average_sources=False)
        assert all(sir == input_sxr(images, noise, average_sources=False).sir)
        np.testing.assert_allclose(sdr, 2 * [10 * np.log10(100/101)], atol=1e-6)
        np.testing.assert_allclose(sir, 2 * [0], atol=1e-6)
        np.testing.assert_allclose(snr, 2 * [20], atol=1e-6)

    def test_output_sxr_more_outputs_than_sources_inf(self):
        sdr, sir, snr = output_sxr(
            morph('kKt->kKt', np.asarray([
                [1 * self.s1, 0 * self.s2, 0 * self.n],
                [0 * self.s1, 1 * self.s2, 0 * self.n]
            ])),
            morph('Kt->Kt', np.asarray([0 * self.n, 0 * self.n, 1 * self.n])),
            average_sources=False,
        )
        np.testing.assert_allclose(sdr, 2 * [np.inf])
        np.testing.assert_allclose(sir, 2 * [np.inf])
        np.testing.assert_allclose(snr, 2 * [np.inf])

    def test_output_sxr_more_outputs_than_sources(self):
        sdr, sir, snr = output_sxr(
            morph('kKt->kKt', np.asarray([
                [10 * self.s1, 1 * self.s2, 0 * self.n],
                [0 * self.s1, 10 * self.s2, 0 * self.n]
            ])),
            morph('Kt->Kt', np.asarray([10 * self.n, 0 * self.n, 0 * self.n])),
            average_sources=False,
        )
        np.testing.assert_allclose(sdr, [0, 20], atol=1e-6)
        np.testing.assert_allclose(sir, [np.inf, 20], atol=1e-6)
        np.testing.assert_allclose(snr, [0, np.inf], atol=1e-6)

    def test_output_sxr(self):
        image_contribution = morph('kKt->kKt', np.asarray([
                [10 * self.s1, 1 * self.s2],
                [0 * self.s1, 10 * self.s2]
            ]))
        noise_contribution = morph('Kt->Kt', np.asarray(
            [10 * self.n, 0 * self.n]
        ))
        sdr, sir, snr = output_sxr(
            image_contribution,
            noise_contribution,
            average_sources=False,
        )
        assert all(snr == output_sxr(
            image_contribution,
            noise_contribution,
            average_sources=False,
        ).snr)
        np.testing.assert_allclose(sdr, [0, 20], atol=1e-6)
        np.testing.assert_allclose(sir, [np.inf, 20], atol=1e-6)
        np.testing.assert_allclose(snr, [0, np.inf], atol=1e-6)


class InputSXRTest(unittest.TestCase):

    @condition.retry(10)
    def test_for_single_input(self):
        expected_snr = np.array(10)
        size = (1, 1, 10000)
        x = np.random.normal(0, 1, size)
        noise = 10 ** (-expected_snr/20) * np.random.normal(0, 1, size[1:])

        SDR, SIR, SNR = input_sxr(x, noise)

        nptest.assert_almost_equal(SDR, expected_snr, decimal=1)
        self.assertTrue(np.isinf(SIR))
        nptest.assert_almost_equal(SNR, expected_snr, decimal=1)

    @condition.retry(10)
    def test_two_inputs_no_noise_no_average(self):
        expected_sir = np.array(10)
        size = (2, 1, 10000)
        x = np.random.normal(0, 1, size)
        x[0, 0, :] *= 10**(expected_sir/20)
        noise = np.zeros((1, 10000))

        SDR, SIR, SNR = input_sxr(x, noise, average_sources=False)

        expected_result = np.array([expected_sir, -expected_sir])
        nptest.assert_almost_equal(SDR, expected_result, decimal=1)
        nptest.assert_almost_equal(SIR, expected_result, decimal=1)
        self.assertTrue(np.isinf(SNR).all())

    @condition.retry(10)
    def test_two_different_inputs_no_noise(self):
        expected_sir = np.array(10)
        size = (2, 1, 10000)
        x = np.random.normal(0, 1, size)
        x[:, 0, 0] *= 10**(expected_sir/20)
        noise = np.zeros((1, 10000))

        SDR, SIR, SNR = input_sxr(x, noise)

        expected_result = np.array(0)
        nptest.assert_almost_equal(SDR, expected_result, decimal=1)
        nptest.assert_almost_equal(SIR, expected_result, decimal=1)
        self.assertTrue(np.isinf(SNR).all())

    @condition.retry(10)
    def test_two_equal_inputs_equal_noise(self):
        size = (2, 1, 10000)
        x = np.random.normal(0, 1, size)
        noise = np.random.normal(0, 1, (1, 10000))

        SDR, SIR, SNR = input_sxr(x, noise)

        nptest.assert_almost_equal(SDR, -3., decimal=1)
        nptest.assert_almost_equal(SIR, 0., decimal=1)
        nptest.assert_almost_equal(SNR, 0., decimal=1)


class OutputSXRTest(unittest.TestCase):

    @condition.retry(10)
    def test_for_single_input(self):
        expected_snr = np.array(10)
        size = (1, 1, 10000)
        x = np.random.normal(0, 1, size)
        noise = 10 ** (-expected_snr/20) * np.random.normal(0, 1, size[1:])

        SDR, SIR, SNR = output_sxr(x, noise)

        nptest.assert_almost_equal(SDR, expected_snr, decimal=1)
        self.assertTrue(np.isinf(SIR))
        nptest.assert_almost_equal(SNR, expected_snr, decimal=1)

    @condition.retry(10)
    def test_two_equal_inputs_no_noise(self):
        expected_sir = np.array(20)
        size = (2, 2, 10000)
        x = np.random.normal(0, 1, size)
        x[0, 1, :] = 10**(-expected_sir/20) * x[0, 0, :]
        x[1, 1, :] = x[0, 1, :]
        x[1, 0, :] = 10**(-expected_sir/20) * x[0, 1, :]
        noise = np.zeros((2, 10000))

        SDR, SIR, SNR = output_sxr(x, noise)

        nptest.assert_almost_equal(SDR, 20., decimal=1)
        nptest.assert_almost_equal(SIR, 20., decimal=1)
        self.assertTrue(np.isinf(SNR).all())

    @condition.retry(10)
    def test_two_equal_inputs_equal_noise(self):
        size = (2, 2, 10000)
        x = np.random.normal(0, 1, size)
        noise = np.random.normal(0, 1, (2, 10000))

        SDR, SIR, SNR = output_sxr(x, noise)

        nptest.assert_almost_equal(SDR, -3., decimal=1)
        nptest.assert_almost_equal(SIR, 0, decimal=1)
        nptest.assert_almost_equal(SNR, 0, decimal=1)