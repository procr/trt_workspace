#!/bin/bash

models=("ResNet50" "VGG19" "AlexNet" "GoogleNet" "InceptionV4" "MobileNet" "MobileNetV2")

for model in  ${models[@]}
do
    echo $mdoel
    ./build/trt_demo $model
done
