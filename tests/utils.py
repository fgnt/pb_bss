import unittest
from dc_integration.utils import is_broadcast_compatible


class TestBroadcastCompatibility(unittest.TestCase):
    def check_true(self, *shapes):
        self.assertTrue(is_broadcast_compatible(*shapes), msg=shapes)

    def check_false(self, *shapes):
        self.assertFalse(is_broadcast_compatible(*shapes), msg=shapes)

    def test(self):
        self.check_true((1, 2, 3), (1, 2, 3))
        self.check_true((3, 1, 3), (3, 3, 3))
        self.check_true((1,), (2,), (2,))

        self.check_false((1, 2, 3), (1, 2, 2))
        self.check_false((1, 2, 3), (1, 2, 3, 4))
        self.check_false((1,), (2,), (3,))

    def test_different_number_of_dimensions(self):
        self.check_true((2, 3), (3,))
        self.check_false((2, 3), (2,))

        self.check_true((1, 2, 3), (2, 3), (3,))
        self.check_false((1, 2, 3), (1, 2), (1,))
