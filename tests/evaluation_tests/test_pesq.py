import unittest
from paderbox.evaluation.pesq_module import pesq
import numpy
import numpy.testing as nptest
from paderbox.io.audioread import audioread
from paderbox.io.data_dir import testing as testing_dir


class TestProposedPESQ(unittest.TestCase):
    """
    This test case was written before the code was adapted.
    This is, why it fails.
    """
    def setUp(self):
        data_dir = testing_dir / 'evaluation' / 'data'
        self.ref_path = data_dir / 'speech.wav'
        self.deg_path = data_dir / 'speech_bab_0dB.wav'

        self.ref_array = audioread(self.ref_path)[0]
        self.deg_array = audioread(self.deg_path)[0]

    def test_wb_scores_with_lists_of_paths_length_one(self):
        scores = pesq(
            [self.ref_path],
            [self.deg_path]
        )
        nptest.assert_equal(scores, numpy.asarray([1.083]))

    def test_wb_scores_with_lists_of_paths_length_two(self):
        scores = pesq(
            [self.ref_path, self.ref_path],
            [self.deg_path, self.ref_path]
        )
        nptest.assert_equal(scores, numpy.asarray([1.083, 4.644]))

    def test_wb_scores_with_lists_of_arrays_length_one(self):
        scores = pesq(
            [self.ref_array],
            [self.deg_array]
        )
        nptest.assert_equal(scores, numpy.asarray([1.083]))

    def test_wb_scores_with_lists_of_arrays_length_two(self):
        scores = pesq(
            [self.ref_array, self.ref_array],
            [self.deg_array, self.ref_array]
        )
        nptest.assert_equal(scores, numpy.asarray([1.083, 4.644]))

    def test_nb_scores_with_lists_of_paths_length_one(self):
        scores = pesq(
            [self.ref_path],
            [self.deg_path],
            'nb'
        )
        nptest.assert_equal(scores, numpy.asarray([1.607]))

    def test_nb_scores_with_lists_of_paths_length_two(self):
        scores = pesq(
            [self.ref_path, self.ref_path],
            [self.deg_path, self.ref_path],
            'nb'
        )
        nptest.assert_equal(scores, numpy.asarray([1.607, 4.549]))

    def test_nb_scores_with_lists_of_arrays_length_one(self):
        scores = pesq(
            [self.ref_array],
            [self.deg_array],
            'nb'
        )
        nptest.assert_equal(scores, numpy.asarray([1.607]))

    def test_nb_scores_with_lists_of_arrays_length_two(self):
        scores = pesq(
            [self.ref_array, self.ref_array],
            [self.deg_array, self.ref_array],
            'nb'
        )
        nptest.assert_equal(scores, numpy.asarray([1.607, 4.549]))

    def test_wb_scores_with_paths_directly(self):
        scores = pesq(
            self.ref_path,
            self.deg_path
        )
        nptest.assert_equal(scores, numpy.asarray([1.083]))

    def test_wrong_file(self):
        with self.assertRaisesRegex(
                ChildProcessError,
                r'An error of type 2  \(Reference or Degraded below 1/4 '
                r'second - processing stopped \) occurred during processing.'
        ):
            pesq(
                __file__,
                self.deg_path
            )
