# This sample uses a Caffe ResNet50 Model to create a TensorRT Inference Engine
import random
from PIL import Image
import numpy as np

import pycuda.driver as cuda
# This import causes pycuda to automatically manage CUDA context creation and cleanup.
import pycuda.autoinit

import tensorrt as trt

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import common

import time

class ModelData(object):
    MODEL_PATH = "ResNet50.caffemodel"
    DEPLOY_PATH = "ResNet50.prototxt"
    INPUT_SHAPE = (3, 224, 224)
    OUTPUT_NAME = "prob"
    # We can convert TensorRT data types to numpy types with trt.nptype()
    DTYPE = trt.float32

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Allocate host and device buffers, and create a stream.
def allocate_buffers(engine):
    # Determine dimensions and create page-locked memory buffers (i.e. won't be swapped to disk) to hold host inputs/outputs.
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(ModelData.DTYPE))
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(ModelData.DTYPE))
    # Allocate device memory for inputs and outputs.
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()
    return h_input, d_input, h_output, d_output, stream

def do_inference(context, h_input, d_input, h_output, d_output, stream):
    # Transfer input data to the GPU.
    cuda.memcpy_htod_async(d_input, h_input, stream)
    # Run inference.
    context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    # Synchronize the stream
    stream.synchronize()

    # Additional test(chenrong06)
    n = 10
    d_in = []
    for i in range(n):
        p = cuda.mem_alloc(h_input.nbytes)
        cuda.memcpy_htod_async(p, h_input, stream)

        start = time.clock()
        context.execute_async(bindings=[int(p), int(d_output)], stream_handle=stream.handle)
        stream.synchronize()
        end = time.clock()
        print("Time used:", end - start)

# The Caffe path is used for Caffe2 models.
def build_engine_caffe(model_file, deploy_file, precision):
    # precision: float, half, int8
    # You can set the logger severity higher to suppress messages (or lower to display more messages).
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.CaffeParser() as parser:
        # Workspace size is the maximum amount of memory available to the builder while building an engine.
        # It should generally be set as high as possible.
        builder.max_workspace_size = common.GiB(1)
        # Load the Caffe model and parse it in order to populate the TensorRT network.
        # This function returns an object that we can query to find tensors by name.
        model_tensors = parser.parse(deploy=deploy_file, model=model_file, network=network, dtype=ModelData.DTYPE)
        # For Caffe, we need to manually mark the output of the network.
        # Since we know the name of the output tensor, we can find it in model_tensors.
        print(model_tensors)
        print(ModelData.OUTPUT_NAME)
        print(model_tensors.find(ModelData.OUTPUT_NAME))
        network.mark_output(model_tensors.find(ModelData.OUTPUT_NAME))

        if precision == "half":
            # enable fp16 (chenrong06)
            builder.fp16_mode = True
            builder.strict_type_constraints = True
            print("pricision: half")
        elif precision == "int8":
            # enable int8 and set quantize (chenrong06)
            # Incomplete version, please refer to workspace/tensorrt/samples/sampleINT8API/sampleINT8API.cpp
            builder.int8_mode = True
            builder.int8_calibrator = None
            builder.strict_type_constraints = True
            print(network.num_layers)
            for i in range(network.num_layers):
                layer = network[i]
                tensor = layer.get_output(0)
                tensor.set_dynamic_range(-1.0, 1.0)
                tensor = layer.get_input(0)
                tensor.set_dynamic_range(-1.0, 1.0)
            print("pricision: int8")
        else:
            print("pricision: float")

        return builder.build_cuda_engine(network)

def load_normalized_test_case(test_image, pagelocked_buffer):
    # Converts the input image to a CHW Numpy array
    def normalize_image(image):
        # Resize, antialias and transpose the image to CHW.
        c, h, w = ModelData.INPUT_SHAPE
        return np.asarray(image.resize((w, h), Image.ANTIALIAS)).transpose([2, 0, 1]).astype(trt.nptype(ModelData.DTYPE)).ravel()

    # Normalize the image and copy to pagelocked memory.
    np.copyto(pagelocked_buffer, normalize_image(Image.open(test_image)))
    return test_image

def main():
    # Set the data path to the directory that contains the trained models and test images for inference.
    #data_path, data_files = common.find_sample_data(description="Runs a ResNet50 network with a TensorRT inference engine.", find_files=[ModelData.MODEL_PATH, ModelData.DEPLOY_PATH])

    data_path, data_files, precision = common.find_sample_data(find_files=[".caffemodel", ".prototxt"])

    # Get test images, models and labels.
    #test_images = data_files[0:3]
    #caffe_model_file, caffe_deploy_file, labels_file = data_files[3:]
    caffe_model_file, caffe_deploy_file = data_files[:]
    #labels = open(labels_file, 'r').read().split('\n')

    # Build a TensorRT engine.
    with build_engine_caffe(caffe_model_file, caffe_deploy_file, precision) as engine:
        # Inference is the same regardless of which parser is used to build the engine, since the model architecture is the same.
        # Allocate buffers and create a CUDA stream.
        h_input, d_input, h_output, d_output, stream = allocate_buffers(engine)
        # Contexts are used to perform inference.
        with engine.create_execution_context() as context:
            # Load a normalized test case into the host input page-locked buffer.
            #test_image = random.choice(test_images)
            #test_case = load_normalized_test_case(test_image, h_input)
            # Run the engine. The output will be a 1D tensor of length 1000, where each value represents the
            # probability that the image corresponds to that label
            do_inference(context, h_input, d_input, h_output, d_output, stream)
            # We use the highest probability as our prediction. Its index corresponds to the predicted label.
            #pred = labels[np.argmax(h_output)]
            #if "_".join(pred.split()) in os.path.splitext(os.path.basename(test_case))[0]:
            #    print("Correctly recognized " + test_case + " as " + pred)
            #else:
            #    print("Incorrectly recognized " + test_case + " as " + pred)

if __name__ == '__main__':
    main()
