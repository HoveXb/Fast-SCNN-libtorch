#include<FastScnn_v3.h>


int main(int argc, const char* argv[]) {
      if (argc != 3) {
    std::cerr << "usage: test <path-to-exported-script-module> <path-to-img>\n";
    return -1;
  }
  FastScnn FastScnn(argv[1],2);
  cv::Mat img = cv::imread(argv[2],cv::IMREAD_COLOR);  
  FastScnn.Inference(img);
  FastScnn.CorlorImg();
}