import sys
sys.path.append('../src')
import unittest
from visualization import *


def dummy_image(i=1):
    return np.random.randint(-1, 1, size=(i, 256, 256, 1))

def dummy_mask(i=1):
    return np.random.randint(0, 1, size=(i, 256, 256, 1))


class TestData(unittest.TestCase):
    def test_output(self):
        i = 1
        output = visualize_output(dummy_image(i), dummy_mask(i))
        self.assertEqual(output.shape, (256 * i, 256 * 2))

        i = 5
        output = visualize_output(dummy_image(i), dummy_mask(i))
        self.assertEqual(output.shape, (256 * i, 256 * 2))


if __name__ == "__main__":
    unittest.main()