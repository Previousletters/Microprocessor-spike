# TVM-SPIKE Work Place

This folder saves whole tvm-spike codes, deep-learning model, image data, and so on.

## img

It stoves whole images which are not separated by different deep learning models.

## model

model = /

  --model-work

    (some codes to convert models)

  --onnx-model

  --tf-model

  --tflite-model

  model-list.py

It stoves all the deep-learning models. Like onnx, tensorflow(tf), tensorflow lite(tflite). And there is a work folder which named "model-work". Actually, it should have pytorch model to test. 

For model-list.py, it is an api python code for the models-path.

## spike

It stoves the main spike--tvm codes. "tvm-spike.py" is the main work code. "tvm-spike-new.py" is also the main work code for tvm0.8(maybe it is very useful). "riscv-spike.py" and so on are original codes.

## temp

Some forgotten files... Maybe they are useful.

## x86

The tvm codes which host target is x86 cpu(our own cpu). It is used to test if the deep learning model could run by tvm code. Then, try to replace x86 by spike.
