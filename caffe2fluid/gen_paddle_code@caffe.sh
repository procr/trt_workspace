#!/bin/bash
#set -x

#models=("ResNet50" "VGG19" "AlexNet" "GoogleNet" "InceptionV4" "MobileNet" "MobileNetV2")
#models=("InceptionV4")
models=("GoogleNet")

for model in  ${models[@]}
do
    echo $model

    prototxt=/trt_workspace/caffe_model/$model.prototxt
    caffemodel=/trt_workspace/caffe_model/$model.caffemodel
    data_out=$model.npy
    code_out=$model.py

    python convert.py $prototxt \
        --caffemodel $caffemodel \
        --data-output-path $data_out \
        --code-output-path $code_out
done
