#include <gflags/gflags.h>
#include <glog/logging.h>
#include "utils.h"
#include <chrono>
#include <string>

namespace paddle {
namespace demo {

inline uint64_t GetTimeInNsec() {
    using clock = std::conditional<std::chrono::high_resolution_clock::is_steady,
          std::chrono::high_resolution_clock,
          std::chrono::steady_clock>::type;
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
            clock::now().time_since_epoch())
        .count();
}

/*
 * Use the tensorrt fluid engine to inference the demo.
 */
void Main(std::string name) {
  std::unique_ptr<PaddlePredictor> predictor;
  paddle::contrib::AnalysisConfig config(true);
  //config.model_dir = "/Paddle/inference.model.resnet50.batch1.pass1.flowers";

  //config.param_file = "/Paddle/demo_trt/resnet50/params";
  //config.prog_file = "/Paddle/demo_trt/resnet50/model";

  //config.param_file = "/models/fluid/PaddleCV/image_classification/output/" + name + "/params";
  //config.prog_file = "/models/fluid/PaddleCV/image_classification/output/" + name + "/model";

  config.param_file = "/trt_workspace/caffe2fluid/fluid/" + name + "/params";
  config.prog_file = "/trt_workspace/caffe2fluid/fluid/" + name + "/model";

  printf("%s\n%s\n", config.param_file.c_str(), config.prog_file.c_str());
  config.device = 0;
  config.EnableTensorRtEngine();
  config.fraction_of_gpu_memory = 0.15;  // set by yourself

  //config.pass_builder()->TurnOnDebug();
  //config.pass_builder()->DeletePass("infer_clean_graph_pass");
  //config.pass_builder()->DeletePass("graph_viz_pass");
  //config.pass_builder()->DeletePass("conv_bn_fuse_pass");
  //config.pass_builder()->DeletePass("graph_viz_pass");
  //config.pass_builder()->DeletePass("fc_fuse_pass");

  predictor = CreatePaddlePredictor(config);

  int n = 10;
  int c = 3;
  int h = 224;
  int w = 224;
  int img_size = c * h * w;
  float data[n * c * h * w];
  int64_t label[n];

  static std::default_random_engine s_generator;
  std::uniform_real_distribution<float> distribution(-100, 100);
  for (size_t i = 0; i < n * c * h * w; ++i) {
      data[i] = distribution(s_generator);
  }
  for (size_t i = 0; i < n; ++i) {
      label[i] = (int64_t)i;
  }

  for (int i = 0; i < n; ++i) {
      PaddleTensor tensor_data;
      tensor_data.shape = std::vector<int>({1, c, h, w});
      tensor_data.data = PaddleBuf(static_cast<void *>(&data[i * img_size]), img_size * sizeof(float));
      tensor_data.dtype = PaddleDType::FLOAT32;

      /*
      PaddleTensor tensor_label;
      tensor_label.shape = std::vector<int>({1, 1000});
      tensor_label.data = PaddleBuf(static_cast<void *>(&label[i]), sizeof(float));
      tensor_label.dtype = PaddleDType::INT64;
      */

      std::vector<PaddleTensor> outputs;

      int64_t t1 = GetTimeInNsec();
      //predictor->Run({tensor_data, tensor_label}, &outputs, 1);
      predictor->Run({tensor_data}, &outputs, 1);
      int64_t t2 = GetTimeInNsec();
      printf("time: %ld ns\n", t2 - t1);
  }
}

}  // namespace demo
}  // namespace paddle

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  if (argc != 2) {
      printf("Please enter model name\n");
      return 0;
  }
  paddle::demo::Main(argv[1]);
  return 0;
}
