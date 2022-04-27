#include "FastScnn.h"
// #include <algorithm>
FastScnn::FastScnn(const std::string& path,const int num_class):mnum_class(num_class){

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
    
    // check if cuda is available
}

FastScnn::~FastScnn()
{
}

cv::Mat FastScnn::Inference(cv::Mat img,int crop_h,int crop_w,float stride_rate)
{
    // cv::Mat imgcopy= img.clone();
    cv::cvtColor(img,img,cv::COLOR_BGR2RGB);
    ori_w = img.cols;
    ori_h = img.rows;

    int pad_h = std::max(crop_h-ori_h,0);
    int pad_w = std::max(crop_w-ori_w,0);
    int pad_half_h = int(pad_h/2);
    int pad_half_w = int(pad_w/2);

    if(pad_h>0 || pad_w>0)
    {
        cv::copyMakeBorder(img,img,pad_half_h,pad_h-pad_half_h,
                            pad_half_w,pad_w-pad_half_w,cv::BORDER_CONSTANT,255);//cv::Scalar(0.485,0.456,0.406));
    }

    int new_h = img.rows;
    int new_w = img.cols;

    // transforms.ToTensor(): transform img from HWC to CHW; transform img value to [0,1]
    // 以下代码用时3-4ms
    cv::Mat normedImg;
    img.convertTo(normedImg,CV_32FC3, 1.f / 255.f, 0);
    auto img_tensor =  torch::from_blob(normedImg.data, {1, new_h, new_w, 3});
    img_tensor = img_tensor.permute({0, 3, 1, 2});
    
    if (cuda_available)
    {
        img_tensor = img_tensor.cuda();
    }

    // transform img value to [0,1] 10^-5s
    img_tensor[0][0].sub_(0.485).div_(0.229);
    img_tensor[0][1].sub_(0.456).div_(0.224);
    img_tensor[0][2].sub_(0.406).div_(0.225);

    
    int stride_h = int(std::ceil(crop_h * stride_rate));
    int stride_w = int(std::ceil(crop_w * stride_rate));
    int grid_h = int(std::ceil(float(new_h - crop_h)/stride_h)+1);
    int grid_w = int(std::ceil(float(new_w - crop_w)/stride_w)+1);
    
    torch::Tensor prediction = torch::zeros({mnum_class,new_h,new_w});
    torch::Tensor img_crop = torch::zeros({1,crop_h,crop_w,3});

    //10^-6s 
    if (cuda_available)
    {
        prediction = prediction.cuda();
        img_crop = img_crop.cuda();
    }

    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;

    int start_h,end_h,start_w,end_w =0;
    for(int index_h=0;index_h<grid_h;index_h++)
    {
        // 为减少前向推理时，生成计算图占用的显存
        // 1.使用torch::tensor 而不是at::tensor;2.对输出结果detach 3. torch::NoGradGuard no_grad;
        torch::NoGradGuard no_grad;
        start_h = index_h*stride_h;
        end_h = std::min(start_h+crop_h,new_h);
        start_h = end_h - crop_h;
        for(int index_w=0;index_w<grid_w;index_w++)
        {
            start_w = index_w*stride_w;
            end_w = std::min(start_w+crop_w,new_w);
            start_w = end_w - crop_w;
            // 以下操作占用2-3ms
            if(cuda_available)
            {
                img_crop = img_tensor.index_select(2,torch::arange(start_h,end_h).cuda());
                img_crop = img_crop.index_select(3,torch::arange(start_w,end_w).cuda());
            }
            else
            {
            img_crop = img_tensor.index_select(2,torch::arange(start_h,end_h));
            img_crop = img_crop.index_select(3,torch::arange(start_w,end_w));
            }
            inputs.clear();
            inputs.push_back(img_crop);

            // output [1,c,h,w] //与分辨率相关，1024x2048大概40ms，256x512 1-2ms
            torch::Tensor output = m_module.forward(inputs).toTuple()->elements()[0].toTensor();
            torch::Tensor crop_result = prediction.slice(1,start_h,end_h);
            crop_result = crop_result.slice(2,start_w,end_w);
            crop_result +=output.squeeze(0);//此处记得加detach 否则会出现内存溢出

            // cv::imshow("crop_img",imgcopy(cv::Rect(start_w,start_h,crop_w,crop_h)));
            // cv::waitKey(-1);
        }
    }
    torch::Tensor prep=prediction.argmax(0).to(torch::kUInt8);
    if(cuda_available)
    {
        prep = prep.index_select(0,torch::arange(pad_half_h,pad_half_h+ori_h).cuda());
        prep = prep.index_select(1,torch::arange(pad_half_w,pad_half_w+ori_w).cuda());
        prep = prep.to(at::kCPU);
    }
    else
    {
        prep = prep.index_select(0,torch::arange(pad_half_h,pad_half_h+ori_h));
        prep = prep.index_select(1,torch::arange(pad_half_w,pad_half_w+ori_w));
    }
    cv::Mat label_img=cv::Mat::ones(cv::Size(ori_w,ori_h), CV_8UC1);
    std::memcpy(label_img.data, prep.data_ptr(), prep.numel() *sizeof(torch::kUInt8));
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
            if(label==11)
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