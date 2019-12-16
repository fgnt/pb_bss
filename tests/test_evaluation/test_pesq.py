import unittest
import numpy as np
import paderbox as pb
from paderbox.testing.testfile_fetcher import get_file_path
from pb_bss.evaluation import pesq


class TestProposedPESQ(unittest.TestCase):
    def setUp(self):
        self.ref_path = get_file_path('speech.wav')
        self.deg_path = get_file_path('speech_bab_0dB.wav')

        self.ref_array = pb.io.load_audio(self.ref_path)
        self.deg_array = pb.io.load_audio(self.deg_path)

    def test_wb_scores_with_lists_of_paths_length_one(self):
        # ToDo: pesq does not support filesnames in the moment
        return
        scores = pesq(
            [self.ref_path],
            [self.deg_path],
            sample_rate=16000,
        )
        np.testing.assert_allclose(scores, np.asarray([1.083]))

    def test_wb_scores_with_lists_of_paths_length_two(self):
        # ToDo: pesq does not support filesnames in the moment
        return
        scores = pesq(
            [self.ref_path, self.ref_path],
            [self.deg_path, self.ref_path],
            sample_rate=16000,
        )
        np.testing.assert_allclose(scores, np.asarray([1.083, 4.644]))

    def test_wb_scores_with_lists_of_arrays_length_one(self):
        scores = pesq(
            [self.ref_array],
            [self.deg_array],
            sample_rate=16000,
        )
        np.testing.assert_allclose(scores, np.asarray([1.083234]), rtol=1e-6)

    def test_wb_scores_with_lists_of_arrays_length_two(self):
        scores = pesq(
            [self.ref_array, self.ref_array],
            [self.deg_array, self.ref_array],
            sample_rate=16000,
        )
        np.testing.assert_allclose(scores, np.asarray([1.083234, 4.643888]), rtol=1e-6)

    def test_nb_scores_with_lists_of_paths_length_one(self):
        # ToDo: pesq does not support filesnames in the moment
        return
        scores = pesq(
            [self.ref_path],
            [self.deg_path],
            sample_rate=16000,
            mode='nb',
        )
        np.testing.assert_allclose(scores, np.asarray([1.607]))

    def test_nb_scores_with_lists_of_paths_length_two(self):
        # ToDo: pesq does not support filesnames in the moment
        return
        scores = pesq(
            [self.ref_path, self.ref_path],
            [self.deg_path, self.ref_path],
            sample_rate=16000,
            mode='nb',
        )
        np.testing.assert_allclose(scores, np.asarray([1.607, 4.549]))

    def test_nb_scores_with_lists_of_arrays_length_one(self):
        scores = pesq(
            [self.ref_array],
            [self.deg_array],
            sample_rate=16000,
            mode='nb',
        )
        np.testing.assert_allclose(scores, np.asarray([1.607208]))

    def test_nb_scores_with_lists_of_arrays_length_two(self):
        scores = pesq(
            [self.ref_array, self.ref_array],
            [self.deg_array, self.ref_array],
            sample_rate=16000,
            mode='nb',
        )
        np.testing.assert_allclose(scores, np.asarray([1.607208, 4.548638]))

    def test_wb_scores_with_paths_directly(self):
        # ToDo: pesq does not support filesnames in the moment
        return
        scores = pesq(
            self.ref_path,
            self.deg_path,
            sample_rate=16000,
        )
        np.testing.assert_allclose(scores, np.asarray([1.083]))

    def test_wrong_file(self):
        # ToDo: pesq does not support filesnames in the moment
        return
        with self.assertRaisesRegex(
                ChildProcessError,
                r'An error of type 2  \(Reference or Degraded below 1/4 '
                r'second - processing stopped \) occurred during processing.'
        ):
            pesq(
                __file__,
                self.deg_path
            )
