import sys
sys.path.append("..")
import models
import keras

a = models.model_list.get_model("CNN.tflite")
print(a)