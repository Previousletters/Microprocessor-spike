import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model("tf-module/model_DNN.h5")
img = cv2.imread("img/Three.jpeg")
img = img[:,:,0]
grey=cv2.resize(img, (28, 28),interpolation=cv2.INTER_CUBIC)
# cv2.imwrite("grey.jpg", grey)
img = 1- grey.reshape([1, 784])/255
y = model.predict_classes(img)

print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
print(grey.shape)
print("The picture is {}".format(y))
plt.axis("off")
plt.imshow(img.reshape([28, 28]), cmap="Greys", interpolation="nearest")
plt.show()