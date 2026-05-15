import unittest
import tensorflow as tf
from src.model import build_residual_attention_unet

class TestModel(unittest.TestCase):
    def test_model_build(self):
        n_classes = 6
        img_height = 256
        img_width = 256
        img_channels = 3
        
        model = build_residual_attention_unet(n_classes, img_height, img_width, img_channels)
        
        self.assertEqual(model.output_shape, (None, img_height, img_width, n_classes))
        self.assertEqual(model.input_shape, (None, img_height, img_width, img_channels))
        print("Model built successfully!")

if __name__ == '__main__':
    unittest.main()
