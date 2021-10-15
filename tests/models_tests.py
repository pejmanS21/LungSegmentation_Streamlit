import sys
sys.path.append('../src')

import unittest
from resunet import ResUnet_Builder
from unet import Unet_Builder
from visualization import visualize_vae
from vae import decoder
import numpy as np


def dummy_image(i=1):
    return np.random.randint(-1, 1, size=(i, 256, 256, 1))


class TestModels(unittest.TestCase):
        
    def test_unet(self):
        model_unet = Unet_Builder(pretrained_weights="../weigths/cxr_seg_unet.hdf5",
                              input_size=(256, 256, 1), pretrained=True)

        i = 1
        output = model_unet.predict(dummy_image(i))
        self.assertEqual(output.shape, (i, 256, 256, 1))
        
        i = 5
        output = model_unet.predict(dummy_image(i))
        self.assertEqual(output.shape, (i, 256, 256, 1))
    
    def test_runet(self):
        model_runet = ResUnet_Builder(pretrained_weights="../weigths/cxr_seg_res_unet.hdf5",
                              input_size=(256, 256, 1), pretrained=True)

        i = 1
        output = model_runet.predict(dummy_image(i))
        self.assertTrue(output.shape == (i, 256, 256, 1))
        
        i = 5
        output = model_runet.predict(dummy_image(i))
        self.assertTrue(output.shape == (i, 256, 256, 1))

    def test_vae(self):
        model_decoder = decoder(pretrained_weights="../weigths/decoder.hdf5", pretrained=True)
        
        num_out, num_range = 3, 4
        output = visualize_vae(model_decoder, num_out, num_range)
        self.assertEqual(output.shape, (256 * num_out, 256 * num_out))

        num_out, num_range = 1, 8
        output = visualize_vae(model_decoder, num_out, num_range)
        self.assertEqual(output.shape, (256 * num_out, 256 * num_out))


if __name__ == "__main__":
    unittest.main()