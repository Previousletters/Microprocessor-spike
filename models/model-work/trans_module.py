from keras.backend import clear_session
import numpy as np
import tensorflow as tf
from tensorflow import keras

clear_session()
np.set_printoptions(suppress=True)
input_graph_name = "tf-module/model_CNN_new.h5"
output_graph_name = 'tflite-module/DNN.tflite'
converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(model_file=input_graph_name)
converter.post_training_quantize = True
tflite_model = converter.convert()
open(output_graph_name, "wb").write(tflite_model)
print ("generate:",output_graph_name)
