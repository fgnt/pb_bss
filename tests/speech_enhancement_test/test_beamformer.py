import unittest
import numpy as np

from nt.speech_enhancement.beamformer import get_gev_vector
from nt.speech_enhancement.beamformer import get_lcmv_vector
from nt.speech_enhancement.beamformer import get_mvdr_vector
from nt.speech_enhancement.beamformer import get_pca_vector


def rand(*shape, data_type=np.complex128):
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
        K, F, D = 2, 3, 6
        self.shapes = [
            ((D, D), (D,)),
            ((F, D, D), (F, D)),
            ((K, F, D, D), (K, F, D)),
        ]

    def test_gev_dimensions(self):
        for shape in self.shapes:
            output = get_gev_vector(rand(shape[0]), rand(shape[0]))
            assert output.shape == shape[1]

    def test_pca_dimensions(self):
        for shape in self.shapes:
            output = get_pca_vector(rand(shape[0]))
            assert output.shape == shape[1]

    def test_mvdr_dimensions(self):
        for shape in self.shapes:
            output = get_mvdr_vector(rand(shape[1]), rand(shape[0]))
            assert output.shape == shape[1]

    def test_lcmv_dimensions(self):
        K, F, D = 2, 3, 6
        output = get_lcmv_vector(rand((K, F, D)), [1, 0], rand((F, D, D)))
        assert output.shape == (F, D)


