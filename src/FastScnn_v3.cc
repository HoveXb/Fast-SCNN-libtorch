#include "FastScnn_v3.h"
// #include <algorithm>
FastScnn::FastScnn(const std::string& path,const int num_class){

    // check if cuda is available
    if (torch::cuda::is_available()) {
    std::cout << "CUDA available! Inference on GPU." << std::endl;
    mdevice_type = torch::kCUDA;//torch::kCUDA;
    cuda_available = true;
    } else {
    std::cout << "Inference on CPU." << std::endl;
    mdevice_type = torch::kCPU;
    cuda_available = false;
    }
    // torch::Device device(mdevice_type);
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        std::cout << "Loading  model" << std::endl;
        m_module = torch::jit::load(path);
        m_module.to(mdevice_type);
        m_module.eval();
        std::cout << "Load model succeedly!" << std::endl;
    }
    catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    }

    mnum_class = num_class;
}

FastScnn::~FastScnn()
{
}

cv::Mat FastScnn::Inference(cv::Mat img,int cropsize)
{
    // cv::Mat imgcopy= img.clone();
    cv::cvtColor(img,img,cv::COLOR_BGR2RGB);
    ori_w = img.cols;
    ori_h = img.rows;
    
    int iw,ih;
    if(ori_w>ori_h)
    {
        iw = cropsize;
        ih = int(1.0*ori_h*iw/ori_w);
        int pad_h = cropsize - ih;
        cv::resize(img,img,cv::Size(iw,ih),0,0,1);
        cv::copyMakeBorder(img,img,0,pad_h,
                            0,0,cv::BORDER_CONSTANT,0);
    }
    else
    {
        ih = cropsize;
        iw = int(1.0*ori_w*ih/ori_h);
        int pad_w = cropsize - iw;
        cv::resize(img,img,cv::Size(iw,ih));
        cv::copyMakeBorder(img,img,0,0,
                            0,pad_w,cv::BORDER_CONSTANT,0);
    }

    int new_h = img.rows;
    int new_w = img.cols;

    // transforms.ToTensor(): transform img from HWC to CHW; transform img value to [0,1]
    cv::Mat normedImg;
    img.convertTo(normedImg,CV_32FC3, 1.f / 255.f, 0);

    auto img_tensor =  torch::from_blob(normedImg.data, {1, new_h, new_w, 3});
    img_tensor = img_tensor.permute({0, 3, 1, 2});

    if (cuda_available)
    {
        img_tensor = img_tensor.cuda();
    }
    // transform img value to [0,1]
    img_tensor[0][0].sub_(0.485).div_(0.229);
    img_tensor[0][1].sub_(0.456).div_(0.224);
    img_tensor[0][2].sub_(0.406).div_(0.225);
    
    // prediction赋值不能少；
    torch::Tensor prediction = torch::zeros({mnum_class,new_h,new_w});
    if (cuda_available)
    {
        prediction = prediction.cuda();
    }
    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    {
        // 为减少前向推理时，生成计算图占用的显存
        // 1.使用torch::tensor 而不是at::tensor;2.对输出结果detach 3. torch::NoGradGuard no_grad;
        torch::NoGradGuard no_grad;
        inputs.clear();
        inputs.push_back(img_tensor);
        torch::Tensor output = m_module.forward(inputs).toTuple()->elements()[0].toTensor();//输出纬度{1,c,h,w}
        prediction = output.squeeze(0);
    }
    torch::Tensor prep=prediction.argmax(0).to(torch::kUInt8);
    if(cuda_available)
    {
        prep = prep.index_select(0,torch::arange(0,ih).cuda());
        prep = prep.index_select(1,torch::arange(0,iw).cuda());
        prep = prep.to(at::kCPU);
    }
    else
    {
        prep = prep.index_select(0,torch::arange(0,ih));
        prep = prep.index_select(1,torch::arange(0,iw));
    }

    cv::Mat label_img=cv::Mat::ones(cv::Size(iw,ih), CV_8UC1);
    std::memcpy(label_img.data, prep.data_ptr(), prep.numel() *sizeof(torch::kUInt8));
    cv::resize(label_img,label_img,cv::Size(ori_w,ori_h),0,0,0);
    mlabel_img = label_img;
    return label_img;
}

void FastScnn::CorlorImg()
{
    cv::Mat coloredImg(ori_h,ori_w, CV_8UC3);

    // size_t min_label=255,max_label=0;
    for(size_t x=0;x<ori_h;x++)
    {
        for(size_t y=0;y<ori_w;y++)
        {
            int label=mlabel_img.at<uint8_t>(x,y);
            // uint8_t label=prep[x][y];
            // std::cout<<int(label)<<std::endl;
            if(label==1)
            {
                coloredImg.at<cv::Vec3b>(x,y)=mcolorMap[label];
            }
            else
            {
                coloredImg.at<cv::Vec3b>(x,y)=cv::Vec3b(0,0,0);
            }
        }
    }
    cv::cvtColor(coloredImg,coloredImg,cv::COLOR_RGB2BGR);
    cv::imshow("result",coloredImg);
    cv::imwrite("../111.png",coloredImg);
    cv::waitKey(-1);
}