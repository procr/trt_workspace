#!/bin/bash

echo $PWD

#sudo docker rm paddle_trt
#sudo nvidia-docker run --name paddle_trt -v $PWD/Paddle:/Paddle -v $PWD/models:/models -it docker.io/xreki/paddle:cuda92_cudnn7 /bin/bash

#sudo docker rm trt_test
#sudo nvidia-docker run -it --name trt_test -v $PWD/Paddle:/Paddle -v $PWD/models:/models -v $PWD/trt_workspace:/trt_workspace -v $PWD/DeepBench:/DeepBench nvcr.io/nvidia/tensorrt:18.11-py3 /bin/bash

#sudo docker rm caffe_test
#sudo nvidia-docker run -it --name caffe_test -v $PWD/Paddle:/Paddle -v $PWD/models:/models -v $PWD/trt_workspace:/trt_workspace -v $PWD/DeepBench:/DeepBench bvlc/caffe:gpu /bin/bash

sudo docker rm paddle
sudo nvidia-docker run --name paddle -v $PWD/Paddle:/Paddle -v $PWD/trt_workspace:/trt_workspace -it procr/paddle_docker:cuda92_cudnn7_trt /bin/bash

#sudo docker rm trt
#sudo nvidia-docker run --name trt -v $PWD/trt_workspace:/trt_workspace -it procr/tensorrt:18.11-py3 /bin/bash

#sudo docker rm caffe
#sudo nvidia-docker run --name caffe -v $PWD/trt_workspace:/trt_workspace -it procr/caffe:gpu /bin/bash
