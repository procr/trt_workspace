#!/bin/bash
#set -x

models=("ResNet50" "VGG19" "AlexNet" "GoogleNet" "InceptionV4" "MobileNet" "MobileNetV2")
precision=("float" "half" "int8")

for m in  ${models[@]}
do
    for p in  ${precision[@]}
    do
        echo $m $p
        python trt_perf.py -model $m -precision $p  -d /trt_workspace/caffe_model/
        echo
    done
done



