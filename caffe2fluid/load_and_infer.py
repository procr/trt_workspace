from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys

import argparse
import functools
import numpy as np
import time

import cProfile, pstats, StringIO

import paddle.v2 as paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.profiler as profiler
from paddle.fluid import debugger

def infer(place, save_dirname=None, trans=False):
    if save_dirname is None:
        return

    exe = fluid.Executor(place)

    inference_scope = fluid.core.Scope()
    with fluid.scope_guard(inference_scope):
        # Use fluid.io.load_inference_model to obtain the inference program desc,
        # the feed_target_names (the names of variables that will be feeded
        # data using feed operators), and the fetch_targets (variables that
        # we want to obtain data from using fetch operators).
        [inference_program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(save_dirname, exe,
                 model_filename='model', params_filename='params')

        assert feed_target_names[0] == 'data'
        #assert feed_target_names[1] == 'label'

        print(feed_target_names)
        print(fetch_targets)

        if (trans):
            inference_transpiler_program = inference_program.clone()
            t = fluid.transpiler.InferenceTranspiler()
            t.transpile(inference_transpiler_program, place)
            prog = inference_transpiler_program
        else:
            prog = inference_program

        """
        for block in inference_program.blocks:
            for op in block.ops:
                print(op.type)

        print("----------------")

        for block in inference_transpiler_program.blocks:
            for op in block.ops:
                print(op.type)

        print(debugger.pprint_program_codes(inference_program))
        print("----------------")
        print(debugger.pprint_program_codes(inference_transpiler_program))
        exit()
        """

        for i in range(10):
            img_data = np.random.random([1, 3, 224, 224]).astype('float32')
            if (i == 9):
                profiler.start_profiler("All")
            exe.run(prog,
                    feed={feed_target_names[0]: img_data}, 
                    fetch_list=fetch_targets)
            if (i == 9):
                profiler.stop_profiler("total", "/tmp/profile")



if __name__ == '__main__':

    if len(sys.argv) != 2:
        print('Please input model name')
        exit(1)
    model = sys.argv[1]
    #model="ResNet50"
    save_dirname = "/trt_workspace/caffe2fluid/fluid/" + model + "/"

    print(model)
    print(save_dirname)

    place = core.CUDAPlace(0)

    #infer(place, save_dirname, trans=True)
    infer(place, save_dirname, trans=False)
