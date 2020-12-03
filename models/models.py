work = "/home/previous/Desktop/Work/tvm-work/models/"
suffix = {
	"h5": "keras-model", 
	"onnx": "onnx-model", 
	"tflite": "tflite-model", 
}
dlmodels = {
	"keras-model": ["model_CNN.h5", "model_CNN_new.h5", "model_DNN.h5"],
	"onnx-model": ["mobilenet_v2.onnx", "model_CNN.onnx", "super_resolution.onnx", "test.onnx"],
	"tflite-model": ["CNN.tflite", "CNN_new.tflite", "DNN.tflite", "mobilenet_v1.tflite", "mobilenet_v1_0.25_128_quant.tflite"],
}