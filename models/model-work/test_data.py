import matplotlib
matplotlib.use("tkagg")

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets
from matplotlib import pyplot as plt
import numpy as np

(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = datasets.mnist.load_data()

print(y_train_raw[0])
print(x_train_raw.shape, y_train_raw.shape)
print(x_test_raw.shape, y_test_raw.shape)

num_classes = 10
y_train = keras.utils.to_categorical(y_train_raw, num_classes)
y_test = keras.utils.to_categorical(y_test_raw, num_classes)

print(y_train[0])
print(y_train.shape, y_test.shape)

plt.figure()
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(x_train_raw[i])
    plt.axis('off')
plt.show()
