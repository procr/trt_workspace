#!/bin/bash
#set -x

#models=("ResNet50" "VGG19" "AlexNet" "GoogleNet" "InceptionV4" "MobileNet" "MobileNetV2")
models=("GoogleNet")

for model in  ${models[@]}
do
    echo $model

    data_out=$model.npy
    code_out=$model.py
    python $code_out $data_out ./fluid/$model/
done
