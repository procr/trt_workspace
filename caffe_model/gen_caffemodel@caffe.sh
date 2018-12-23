#!/usr/bin/env sh
set -e
set -x

/opt/caffe/build/tools/caffe train --solver=solver.prototxt $@
#/opt/caffe/build/tools/caffe time --model=ResNet50.prototxt --weights=_iter_1.caffemodel -gpu 0
