import unittest
import numpy as np
from paderbox.speech_enhancement.beamformer import gev_wrapper_on_masks
from paderbox.speech_enhancement.beamformer import pca_wrapper_on_masks
from paderbox.speech_enhancement.beamformer import pca_mvdr_wrapper_on_masks


def rand(*shape, data_type):
    if not shape:
        shape = (1,)
    elif isinstance(shape[0], tuple):
        shape = shape[0]

    def uniform(data_type_local):
        return np.random.uniform(-1, 1, shape).astype(data_type_local)

    if data_type in (np.float32, np.float64):
        return uniform(data_type)
    elif data_type is np.complex64:
        return uniform(np.float32) + 1j * uniform(np.float32)
    elif data_type is np.complex128:
        return uniform(np.float64) + 1j * uniform(np.float64)


class TestBeamformerWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        frames, sensors, bins = 100, 3, 513
        self.mix = rand(frames, sensors, bins, data_type=np.complex128)
        self.target_mask = np.random.uniform(0, 1, (frames, bins))
        self.noise_mask = np.random.uniform(0, 1, (frames, bins))
        self.output_shape = (frames, bins)

    def test_gev_with_target_mask(self):
        output = gev_wrapper_on_masks(
            mix=self.mix,
            target_mask=self.target_mask,
            noise_mask=None,
            normalization=None
        )
        assert output.shape == self.output_shape

    def test_gev_with_noise_mask(self):
        output = gev_wrapper_on_masks(
            mix=self.mix,
            target_mask=None,
            noise_mask=self.noise_mask,
            normalization=None
        )
        assert output.shape == self.output_shape

    def test_gev_with_both_masks(self):
        output = gev_wrapper_on_masks(
            mix=self.mix,
            target_mask=self.target_mask,
            noise_mask=self.noise_mask,
            normalization=None
        )
        assert output.shape == self.output_shape

    def test_gev_with_both_masks_and_normalization(self):
        output = gev_wrapper_on_masks(
            mix=self.mix,
            target_mask=self.target_mask,
            noise_mask=self.noise_mask,
            normalization=True
        )
        assert output.shape == self.output_shape

    def test_pca_with_target_mask(self):
        output = pca_wrapper_on_masks(
            mix=self.mix,
            target_mask=self.target_mask,
            noise_mask=None
        )
        assert output.shape == self.output_shape

    def test_pca_with_noise_mask(self):
        output = pca_wrapper_on_masks(
            mix=self.mix,
            target_mask=None,
            noise_mask=self.noise_mask
        )
        assert output.shape == self.output_shape

    def test_pca_with_both_masks(self):
        output = pca_wrapper_on_masks(
            mix=self.mix,
            target_mask=self.target_mask,
            noise_mask=self.noise_mask
        )
        assert output.shape == self.output_shape

    def test_pca_mvdr_with_target_mask(self):
        output = pca_mvdr_wrapper_on_masks(
            mix=self.mix,
            target_mask=self.target_mask,
            noise_mask=None,
            regularization=None
        )
        assert output.shape == self.output_shape

    def test_pca_mvdr_with_noise_mask(self):
        output = pca_mvdr_wrapper_on_masks(
            mix=self.mix,
            target_mask=None,
            noise_mask=self.noise_mask,
            regularization=None
        )
        assert output.shape == self.output_shape

    def test_pca_mvdr_with_both_masks(self):
        output = pca_mvdr_wrapper_on_masks(
            mix=self.mix,
            target_mask=self.target_mask,
            noise_mask=self.noise_mask,
            regularization=None
        )
        assert output.shape == self.output_shape

    def test_pca_mvdr_with_both_masks_and_normalization(self):
        output = pca_mvdr_wrapper_on_masks(
            mix=self.mix,
            target_mask=self.target_mask,
            noise_mask=self.noise_mask,
            regularization=1e-5
        )
        assert output.shape == self.output_shape
