import sys
sys.path.append('../src')
import unittest
from data import *


def dummy_image(i=1):
    return np.random.randint(-1, 1, size=(i, 256, 256, 1))


class TestData(unittest.TestCase):
    def test_data(self):

        image = get_data("../images/00000165_008.png", dim=256, pre_process=True)
        self.assertEqual(image.shape, (1, 256, 256, 1))
        self.assertTrue(image.min() == -1)
        
        image = get_data("../images/00000099_003.png", dim=512, pre_process=True)
        self.assertEqual(image.shape, (1, 512, 512, 1))
        self.assertTrue(image.min() == -1)

        for i in range(1, 5 + 1):
            image = get_data("../images", dim=256, n_samples=i)
            self.assertEqual(image.shape, (i, 256, 256, 1))
            self.assertTrue(image.min() == -1)


if __name__ == "__main__":
    unittest.main()