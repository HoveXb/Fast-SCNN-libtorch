#ifndef FastScnn_H
#define FastScnn_H

#include <torch/script.h> 
#include <torch/torch.h>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

class FastScnn
{
public:
    FastScnn(const std::string& weightpath,const int num_class);
    ~FastScnn();
    cv::Mat Inference(cv::Mat img,int cropsize=512);
    void CorlorImg();
private:
    torch::jit::script::Module m_module;

    torch::DeviceType mdevice_type;

    cv::Mat mlabel_img;
    // the origin image size
    int ori_w;
    int ori_h;
    int mnum_class;
    bool cuda_available = false;

    const cv::Vec3b mcolorMap[19]=
    {
        cv::Vec3b(128, 64,128),//'road'
        cv::Vec3b(244, 35,232),//'sidewalk' 
        cv::Vec3b( 70, 70, 70),//building
        cv::Vec3b(102,102,156),//wall
        cv::Vec3b(190,153,153),//fence

        cv::Vec3b(153,153,153),//pole
        cv::Vec3b(250,170, 30),//traffic light
        cv::Vec3b(220,220,  0),//traffic sign
        cv::Vec3b(107,142, 35),//vegetation
        cv::Vec3b(152,251,152),//terrain

        cv::Vec3b( 0,130,180),//sky
        cv::Vec3b(220, 20, 60),//person
        cv::Vec3b(255,  0,  0),//rider
        cv::Vec3b(  0,  0,142),//car
        cv::Vec3b(  0,  0, 70),//truck

        cv::Vec3b(  0, 60,100),//bus
        cv::Vec3b(  0, 80,100),//train
        cv::Vec3b(  0,  0,230),//motorcycle
        cv::Vec3b(119, 11, 32),//bicycle
    };

};

#endif