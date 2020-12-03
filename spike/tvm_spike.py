import os
import sys
sys.path.append("..")
import models
import numpy as np
import tvm
import cv2
from PIL import Image
from tvm import te
from tvm.contrib import graph_runtime, util
from tvm import relay
import tvm.micro as micro
from tvm.micro import create_micro_mod
from tvm.micro.device.base import MemConstraint
from tvm.contrib.download import download_testdata

# Use the riscv_spike emulated micro device.
constraints = {
	'text': (220000, MemConstraint.ABSOLUTE_BYTES),
	'rodata': (256, MemConstraint.ABSOLUTE_BYTES),
	'data': (128, MemConstraint.ABSOLUTE_BYTES),
	'bss': (2048, MemConstraint.ABSOLUTE_BYTES),
	'args': (4096, MemConstraint.ABSOLUTE_BYTES),
	'heap': (100.0, MemConstraint.WEIGHT),
	'workspace': (164000, MemConstraint.ABSOLUTE_BYTES),
	'stack': (32, MemConstraint.ABSOLUTE_BYTES),
}
DEV_RISCV = micro.device.riscv_spike.generate_config(0x10000000,
													 0x7000000,
													 "127.0.0.1",
													 6666,
													 section_constraints=constraints)
TARGET = 'c -device=micro_dev'

def load_model(model_path, shape_dict):
	input_name = list(shape_dict.keys())[0]
	model_type = model_path.split(".")[-1]

	if model_type == 'onnx':
	    import onnx

	    onnx_model = onnx.load(model_path)
	    # return = mod, params
	    return relay.frontend.from_onnx(onnx_model, shape_dict)
	elif model_type == 'pb':
		import tensorflow as tf

	elif model_type == 'tflite':
		dtype_dict = {input_name: 'float32'}
		tflite_model_buf = open(model_path, "rb").read()

	    # Using the buffer, transform into a tflite model python object
		try:
			import tflite
			tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
		except AttributeError:
			import tflite.Model
			tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

		return relay.frontend.from_tflite(tflite_model, shape_dict=shape_dict, dtype_dict=dtype_dict)


def spike_model(model_path, input_x, input_name):
    """Test a program which uses the graph runtime."""
    if not tvm.runtime.enabled("micro_dev"):
        print("not enable micro_dev")
        return

    input_x = input_x.astype("float32")
    shape_dict = {input_name: input_x.shape}
    mod, params = load_model(model_path, shape_dict)

    with micro.Session(DEV_RISCV):
        ctx = tvm.micro_dev(0)

        disable_vectorize = tvm.target.build_config(disable_vectorize=True)
        disable_fusion = relay.build_config(disabled_pass={'FuseOps'})
        with disable_vectorize:
            graph, c_mod, params = relay.build(mod, target=TARGET, params=params)
        print("Part Success")
        micro_mod = micro.create_micro_mod(c_mod, DEV_RISCV)
        mod = graph_runtime.create(graph, micro_mod, ctx)
        mod.set_input(**params)
        mod.set_input(input_name, tvm.nd.array(input_x))
        mod.run
        tvm_output = mod.get_output(0).asnumpy()
        return tvm_output


def test_model(model_path, input_x, input_name):
	target = "llvm"

	shape_dict = {input_name: input_x.shape}
	mod, params = load_model(model_path, shape_dict)
	with tvm.transform.PassContext(opt_level=1):
	    intrp = relay.build_module.create_executor("graph", mod, tvm.cpu(0), target)

	######################################################################
	# Execute on TVM
	# ---------------------------------------------
	dtype = "float32"
	tvm_output = intrp.evaluate()(tvm.nd.array(x.astype(dtype)), **params).asnumpy()
	return tvm_output
	# print('\n\n\n\n\n\n\n\n\n\n\n\ntvm output: ', np.argmax(tvm_output))


def preprocess_image(image_file):
    resized_image = Image.open(image_file).resize((224, 224))
    image_data = np.asarray(resized_image).astype("float32")
    # convert HWC to CHW
    # image_data = image_data.transpose((2, 0, 1))
    # after expand_dims, we have format NCHW
    image_data = np.expand_dims(image_data, axis = 0)
    image_data[:,:,:,0] = 2.0 / 255.0 * image_data[:,:,:,0] - 1 
    image_data[:,:,:,1] = 2.0 / 255.0 * image_data[:,:,:,1] - 1
    image_data[:,:,:,2] = 2.0 / 255.0 * image_data[:,:,:,2] - 1
    return image_data


def post_process(tvm_output, label_file):
    # map id to 1001 classes
    labels = dict()
    with open(label_file) as f:
        for id, line in enumerate(f):
            labels[id] = line
    # convert result to 1D data
    predictions = np.squeeze(tvm_output)
    # get top 1 prediction
    prediction = np.argmax(predictions)

    # convert id to class name
    print("The image prediction result is: id " + str(prediction) + " name: " + labels[prediction])


# model_path = "/home/previous/Desktop/Work/onnx-test/mobilenet_v2.onnx"
# model_path = "/home/previous/Desktop/Work/tflite-workplace/movilenet.tflite"
# model_path = '/home/previous/.tvm_test_data/data/sine_model.tflite'

# Pre data processing
img_path = "../img/seven.png"
img = cv2.imread(img_path)
img = cv2.resize(img, (28, 28))
x = img[np.newaxis, :, :, 0]
# x = x.transpose([0,3,1,2])
x = x[np.newaxis, :, :, :]
model_path = models.model_list.get_model("CNN_new.tflite")
print(x.shape)

if __name__ == "__main__":
	import time
	t0 = time.time()
	tvm_out = spike_model(model_path, x, "input")
	# post_process(tvm_out, "labels.txt")
	print(tvm_out)
	print(time.time()-t0, " seconds")
	print(np.argmax(np.array(tvm_out)))
	print('Finished model test, congratulation!')