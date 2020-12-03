import keras
import keras2onnx
import onnx
from keras.models import load_model
model = load_model('tf-module/model_CNN.h5')  
onnx_model = keras2onnx.convert_keras(model, model.name)
temp_model_file = 'onnx-module/model_CNN.onnx'
onnx.save_model(onnx_model, temp_model_file)
print("Success!")