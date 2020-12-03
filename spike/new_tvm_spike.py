import os

import numpy as np
import tvm
from tvm import te
from tvm.contrib import graph_runtime, util
from tvm import relay
import tvm.micro as micro
from tvm.micro import create_micro_mod
from tvm.contrib.download import download_testdata


# Use the riscv_spike emulated micro device.
DEV_RISCV = micro.device.riscv_spike.default_config(0x10000000, "127.0.0.1", 6666)
TARGET = 'c -device=micro_dev'

def relay_micro_build(func, dev_config, params=None):
    """Create a graph runtime module with a micro device context from a Relay function.

    Parameters
    ----------
    func : relay.Function
        function to compile

    dev_config : Dict[str, Any]
        MicroTVM config dict for the target device

    params : dict
        input parameters that do not change during inference

    Return
    ------
    mod : tvm.runtime.Module
        graph runtime module for the target device
    """
    disable_vectorize = tvm.target.build_config(disable_vectorize=True)
    disable_fusion = relay.build_config(disabled_pass={'FuseOps'})
    with disable_vectorize, disable_fusion:
        graph, c_mod, params = relay.build(func, target=TARGET, params=params)
    micro_mod = micro.create_micro_mod(c_mod, dev_config)
    ctx = tvm.micro_dev(0)
    mod = graph_runtime.create(graph, micro_mod, ctx)
    mod.set_input(**params)
    return mod


def test_alloc():
    """Test tensor allocation on the device."""
    if not tvm.runtime.enabled("micro_dev"):
        return
    shape = (1024,)
    dtype = "float32"

    print("Begin to enter session")
    with micro.Session(DEV_RISCV):
        print("Enter session")
        ctx = tvm.micro_dev(0)
        np_tensor = np.random.uniform(size=shape).astype(dtype)
        micro_tensor = tvm.nd.array(np_tensor, ctx)
        tvm.testing.assert_allclose(np_tensor, micro_tensor.asnumpy())


def test_model():
    """Test a program which uses the graph runtime."""
    if not tvm.runtime.enabled("micro_dev"):
        print("not enable micro_dev")
        return

    model_url = 'https://people.linaro.org/~tom.gall/sine_model.tflite'
    model_file = 'sine_model.tflite'
    model_path = download_testdata(model_url, model_file, module='data')

    tflite_model_buf = open(model_path, "rb").read()

    # Using the buffer, transform into a tflite model python object
    try:
        import tflite
        tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
    except AttributeError:
        import tflite.Model
        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)
   
    # Print out the version of the model
    version = tflite_model.Version()
    print ("Model Version: " + str(version))
    
    input_tensor = "dense_4_input"
    input_shape = (1,)
    input_dtype = "float32"
    mod, params = relay.frontend.from_tflite(tflite_model,
                                            shape_dict={input_tensor: input_shape},
                                            dtype_dict={input_tensor: input_dtype})
    with micro.Session(DEV_RISCV):
        ctx = tvm.micro_dev(0)

        disable_vectorize = tvm.target.build_config(disable_vectorize=True)
        disable_fusion = relay.build_config(disabled_pass={'FuseOps'})
        with disable_vectorize, disable_fusion:
            graph, c_mod, params = relay.build(mod, target=TARGET, params=params)
        
        micro_mod = micro.create_micro_mod(c_mod, DEV_RISCV)
        mod = graph_runtime.create(graph, micro_mod, ctx)
        mod.set_input(**params)
        mod.set_input(input_tensor, tvm.nd.array(np.array([0.5], dtype="float32")))

        tvm_output = mod.get_output(0).asnumpy()
        print("result is: "+str(tvm_output))


def test_conv2d():
    if not tvm.runtime.enabled("micro_dev"):
        return

    from tvm.relay import create_executor
    from tvm.relay import transform

    dshape = (1, 4, 16, 16)
    dtype = 'int8'
    func_name = 'fused_nn_conv2d'

    # reset_gdbinit()

    # Construct Relay program.
    x = relay.var("x", shape=dshape, dtype=dtype)
    conv_expr = relay.nn.conv2d(
            x, relay.var("w"),
            kernel_size=(3, 3),
            padding=(1, 1),
            channels=4)
    func = relay.Function(relay.analysis.free_vars(conv_expr), conv_expr)
    mod = tvm.IRModule.from_expr(func)
    mod = transform.InferType()(mod)

    x_shape = list(map(lambda x: x.value, mod['main'].params[0].checked_type.shape))
    w_shape = list(map(lambda x: x.value, mod['main'].params[1].checked_type.shape))
    out_shape = list(map(lambda x: x.value, mod['main'].ret_type.shape))

    with tvm.target.build_config(disable_vectorize=True):
        graph, c_mod, params = relay.build(mod, target="c")

    with micro.Session(DEV_RISCV):
        micro_mod = micro.create_micro_mod(c_mod, DEV_RISCV)
        candidate_func_name = func_name
        for i in range(100):
            try:
                micro_func = micro_mod[candidate_func_name]
                break
            except tvm.TVMError as e:
                candidate_func_name = f'{func_name}_{i}'
        else:
            assert False
        ctx = tvm.micro_dev(0)

        x_data = tvm.nd.array(np.random.uniform(size=x_shape).astype(dtype), ctx)
        w_data = tvm.nd.array(np.random.uniform(size=w_shape).astype(dtype), ctx)
        result = tvm.nd.array(np.zeros(shape=out_shape, dtype=dtype), ctx)
        micro_func(x_data, w_data, result)

        out_data = np.zeros(out_shape, dtype=dtype)
        params = { 'x': x_data.asnumpy(), 'w': w_data.asnumpy() }
        intrp = create_executor('debug')
        expected_result = intrp.evaluate(mod['main'])(x_data, w_data)

        tvm.testing.assert_allclose(result.asnumpy(), expected_result.asnumpy())


if __name__ == "__main__":
    test_model()
    print()
    print('finished model test')
    input('[press enter to continue]')
    test_alloc()
    print()
    print('finished alloc test')
    input('[press enter to continue]')
    test_conv2d()
    print()
    print('finished conv2d test')
    input('[press enter to continue]')
