set -x
PADDLE_ROOT=/Paddle
TURN_ON_MKL=OFF
TEST_GPU_CPU=ON
TENSORRT_INCLUDE_DIR=/usr/include
TENSORRT_LIB_DIR=/usr/lib
inference_install_dir=${PADDLE_ROOT}/build/fluid_inference_install_dir

USE_TENSORRT=OFF
if [ -d "$TENSORRT_INCLUDE_DIR" -a -d "$TENSORRT_LIB_DIR" ]; then
  USE_TENSORRT=ON
fi

mkdir -p build
cd build

# --------tensorrt------
if [ $USE_TENSORRT == ON -a $TEST_GPU_CPU == ON ]; then
    rm -rf *
    cmake .. -DPADDLE_LIB=${inference_install_dir} \
        -DWITH_MKL=$TURN_ON_MKL \
        -DDEMO_NAME=trt_demo \
        -DWITH_GPU=$TEST_GPU_CPU \
        -DWITH_STATIC_LIB=ON \
        -DUSE_TENSORRT=$USE_TENSORRT \
        -DTENSORRT_INCLUDE_DIR=$TENSORRT_INCLUDE_DIR \
        -DTENSORRT_LIB_DIR=$TENSORRT_LIB_DIR
    make -j
fi
set +x
