import numpy as np
from typing import Tuple
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import mlflow
import mlflow.tensorflow
from base_classes import SegmentationModel



class TFLiteModel(SegmentationModel):
    """
    Reduced Tensorflow Lite Detection Model.
    """

    def __init__(self, model_path: str):
        """
        Initiate the model session.
        """
        self.interpreter = self.__load_model(model_path)

    def __load_model(self, model_path: str) -> tf.lite.Interpreter:
        """
        load the tflite model.
        :param model_path: where to load the model from.
        """
        # Load the TFLite model and allocate tensors.
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        # Get input and output tensors.
        self.input_details = interpreter.get_input_details()
        # print(self.input_details)
        self.output_details = interpreter.get_output_details()
        # print(self.output_details)
        # Test the model on random input data.
        input_shape = self.input_details[0]['shape']
        # print(f'input shape is: {input_shape}')
        return interpreter

    def predict(self, images):
        """
        Use the loaded model to make an estimation.
        :param images: photo to make the prediction on.
        :return: predicted mask.
        """
        images = np.array(images, dtype=np.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], images)
        self.interpreter.invoke()
        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output_data
