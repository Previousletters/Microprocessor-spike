# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Compile ONNX Models
===================
**Author**: `Joshua Z. Zhang <https://zhreshold.github.io/>`_

This article is an introductory tutorial to deploy ONNX models with Relay.

For us to begin with, ONNX package must be installed.

A quick solution is to install protobuf compiler, and

.. code-block:: bash

    pip install onnx --user

or please refer to offical site.
https://github.com/onnx/onnx
"""
import onnx
import numpy as np
import tvm
from tvm import te
import matplotlib.pyplot as plt
import tvm.relay as relay
from tvm.contrib.download import download_testdata

######################################################################
# Load pretrained ONNX model
# ---------------------------------------------
# The example super resolution model used here is exactly the same model in onnx tutorial
# http://pytorch.org/tutorials/advanced/super_resolution_with_caffe2.html
# we skip the pytorch model construction part, and download the saved onnx model
model_path = "model_CNN.onnx"
# now you have super_resolution.onnx on disk
onnx_model = onnx.load(model_path)

######################################################################
# Load a test image
# ---------------------------------------------
# A single cat dominates the examples!
from PIL import Image
import cv2

img_path = "img/Three.jpeg"

img = cv2.imread(img_path)
img = img[:,:,0]
grey=cv2.resize(img, (28, 28),interpolation=cv2.INTER_CUBIC)
# cv2.imwrite("grey.jpg", grey)
x = 1- grey.reshape([1, 28, 28, 1])/255
print('x shape: ',x.shape)

######################################################################
# Compile the model with relay
# ---------------------------------------------
target = "llvm"
# target = "c -device=micro_dev"

input_name = "conv2d_input"
shape_dict = {input_name: x.shape}
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

with tvm.transform.PassContext(opt_level=1):
    intrp = relay.build_module.create_executor("graph", mod, tvm.cpu(0), target)

######################################################################
# Execute on TVM
# ---------------------------------------------
dtype = "float32"
tvm_output = intrp.evaluate()(tvm.nd.array(x.astype(dtype)), **params).asnumpy()
print('\n\n\n\n\n\n\n\n\n\n\n\ntvm output: ', np.argmax(tvm_output))
plt.axis("off")
plt.imshow(cv2.resize(cv2.imread(img_path), (100, 100)), cmap="Greys", interpolation="nearest")
plt.show()