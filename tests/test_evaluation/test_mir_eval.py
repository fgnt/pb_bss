import numpy as np
import unittest

from pb_bss.evaluation.module_mir_eval import mir_eval_sources


class TestMirEval(unittest.TestCase):
    def setUp(self):
        samples = 8000
        self.s1 = np.random.normal(size=(samples,))
        self.s2 = np.random.normal(size=(samples,))
        self.n = np.random.normal(size=(samples,))

    @staticmethod
    def check(reference, estimation, reference_permutation, shape=None):
        sdr, sir, sar, permutation = mir_eval_sources(reference, estimation)
        for sxr in (sdr, sir, sar):
            for value in sxr:
                assert (value > 100).all(), value

        assert np.all(np.equal(reference_permutation, permutation))
        if shape is not None:
            msg = (sdr.shape, sir.shape, sar.shape, permutation.shape)
            assert sdr.shape == shape, msg
            assert sir.shape == shape, msg
            assert sar.shape == shape, msg
            assert permutation.shape == shape, msg

    def test_mir_eval(self):
        self.check(
            np.stack([self.s1, self.s2]),
            np.stack([self.s1, self.s2]),
            reference_permutation=[0, 1],
        )

    def test_mir_eval_swap(self):
        self.check(
            np.stack([self.s1, self.s2]),
            np.stack([self.s2, self.s1]),
            reference_permutation=[1, 0],
        )

    def test_mir_eval_noise_class(self):
        self.check(
            np.stack([self.s1, self.s2]),
            np.stack([self.s2, self.n, self.s1]),
            reference_permutation=[2, 0],
        )

    def test_mir_eval_with_channel(self):
        # Shape: source channel sample
        self.check(
            np.array([
                [self.s1, self.s1, self.s1, self.s1],
                [self.s2, self.s2, self.s2, self.s2],
            ]),
            np.array([
                [self.s2, self.n, self.n, self.n],
                [self.n, self.s2, self.s2, self.s2],
                [self.s1, self.s1, self.s1, self.s1],
            ]),
            reference_permutation=np.array([[2, 0], [2, 1], [2, 1], [2, 1]]).T,
            shape=(2, 4),
        )
