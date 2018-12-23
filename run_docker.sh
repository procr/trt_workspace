#!/bin/bash


sudo docker rm paddle
sudo nvidia-docker run --name paddle -v $PWD/Paddle:/Paddle -v $PWD/trt_workspace:/trt_workspace -w /trt_workspace -it procr/paddle_docker:cuda92_cudnn7_trt /bin/bash

#sudo docker rm trt
#sudo nvidia-docker run --name trt -v $PWD/trt_workspace:/trt_workspace -w /trt_workspace -it procr/tensorrt:18.11-py3 /bin/bash

#sudo docker rm caffe
#sudo nvidia-docker run --name caffe -v $PWD/trt_workspace:/trt_workspace -w /trt_workspace -it procr/caffe:gpu /bin/bash
