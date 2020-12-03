import tensorflow as tf
import matplotlib
matplotlib.use("tkagg")
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets
from matplotlib import pyplot as plt
import numpy as np

(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = datasets.mnist.load_data()

x_test = x_test_raw.reshape(10000, 784)
x_test = x_test.astype('float32')/255

X_test = x_test.reshape(10000, 28, 28, 1)

new_model = keras.models.load_model("tf-module/model_CNN_new.h5")

# matplotlib inline
def res_Visual(n):
    final_opt_a = new_model.predict_classes(X_test[0:n])
    fig, ax = plt.subplots(nrows=int(n/5), ncols=5)
    ax = ax.flatten()
    print("The {} answers are ".format(n))
    print(final_opt_a)
    for i in range(n):
        print(final_opt_a[i], end=",")
        if int((i+1)%5) == 0:
            print("\t")
        img=X_test[i].reshape((28, 28))
        plt.axis("off")
        ax[i].imshow(img, cmap="Greys", interpolation="nearest")
        ax[i].axis("off")
    print("The {} pictures are ".format(n))
    plt.show()

res_Visual(20)
