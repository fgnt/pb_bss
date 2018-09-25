import numpy as np
import unittest
from dc_integration.distribution import GCACGMMTrainer


class TestGCACGMM(unittest.TestCase):
    def setUp(self):
        observation_shape = (15, 100, 3)
        self.observation \
            = np.random.normal(size=observation_shape) \
            + 1j * np.random.normal(size=observation_shape)

        embedding_shape = (15, 100, 10)
        self.embedding = np.random.normal(size=embedding_shape)

    def check_weight(self, weight_constant_axis, expected_weight_shape):
        model = GCACGMMTrainer().fit(
            self.observation,
            self.embedding,
            num_classes=2,
            weight_constant_axis=weight_constant_axis
        )
        self.assertEqual(np.shape(model.weight), expected_weight_shape)

    def test_gcacgmm_no_weight(self):
        self.check_weight((-3, -2, -1), ())

    def test_gcacgmm_weight_k(self):
        self.check_weight((-3, -1), (2,))

    def test_gcacgmm_weight_fk(self):
        self.check_weight((-1,), (15, 2))

    def test_gcacgmm_weight_kt(self):
        self.check_weight((-3,), (2, 100))
