
#include "convolution.h"
#include "../ncnn/platform.h"
#include "../ncnn/layer_type.h"
#include "fused_activation.h"
#include "assert.h"
#ifndef NCNN_TOOLS
#include "../hw/hw_gemm.h"
#include "../hw/hw_sesr_image.h"
#include "../utils/conv_utils.h"
#endif
#include "../hw/accel_params.h"
#include "../utils/utils.h"
#include "../utils/trace.h"
#include <vector>

#ifdef LOG_WGT_BIAS
    int conv_count = 0;
#endif

namespace ncnn {

#ifndef NCNN_TOOLS
static int ncnn_activation_type_to_hw(int activation_type)
{
    switch (activation_type)
    {
    case 0:
        return HW_NONE;
    case 1:
        return HW_RELU;
    case 2:
        return HW_LEAKYRELU;
    case 3:
        return HW_CLIP;
    case 4:
        return HW_SIGMOID;
    case 6:
        return HW_HARDSWISH;
    case 7:
        return HW_TANH;
    case 8:
        return HW_SWISH;
    case 9:
        return HW_ELU;
    case 10:
        return HW_SELU;
    default:
        return activation_type;
    }
}

static float activation_param_or_zero(const Mat& activation_params, int index)
{
    return activation_params.total() > (size_t)index ? activation_params[index] : 0.f;
}
#endif

Convolution::Convolution()
{
    one_blob_only = true;
    support_inplace = false;
}

int Convolution::load_param(const ParamDict& pd)
{
    // printf("[ log ]: load param\n");
    num_output = pd.get(0, 0);
    kernel_w = pd.get(1, 0);
    kernel_h = pd.get(11, kernel_w);
    dilation_w = pd.get(2, 1);
    dilation_h = pd.get(12, dilation_w);
    stride_w = pd.get(3, 1);
    stride_h = pd.get(13, stride_w);
    pad_left = pd.get(4, 0);
    pad_right = pd.get(15, pad_left);
    pad_top = pd.get(14, pad_left);
    pad_bottom = pd.get(16, pad_top);
    pad_value = pd.get(18, 0.f);
    bias_term = pd.get(5, 0);
    weight_data_size = pd.get(6, 0);
    int8_scale_term = pd.get(8, 0);
    activation_type = pd.get(9, 0);
    activation_params = pd.get(10, Mat());
    dynamic_weight = pd.get(19, 0);

    layout_en = pd.get(20, 1);
    ifm_w = pd.get(21, 0);
    ifm_h = pd.get(22, 0);
    ifm_c = pd.get(23, 0);

    if (dynamic_weight)
    {
        one_blob_only = false;
    }

    if (int8_scale_term)
    {
#if NCNN_INT8
        support_int8_storage = true;
#else
        NCNN_LOGE("please build ncnn with NCNN_INT8 enabled for int8 inference");
        return -1;
#endif
    }

    return 0;
}

void Convolution::set_fused_relu()
{
    activation_type = 1;
    activation_params = Mat(2);
    activation_params[0] = 0.f;
    activation_params[1] = 0.f;
}

bool Convolution::can_fuse_relu_on_npu() const
{
    if (dynamic_weight || kernel_w <= 0 || kernel_h <= 0)
        return false;

    int input_channels = ifm_c;
    int kernel_size = kernel_w * kernel_h;
    int weight_size_per_input_channel = kernel_size * num_output;
    if (input_channels <= 0 && weight_size_per_input_channel > 0 && weight_data_size % weight_size_per_input_channel == 0)
        input_channels = weight_data_size / weight_size_per_input_channel;
    if (input_channels <= 0)
        return false;

    int weight_size = kernel_size * u_align(input_channels, 32);
    return weight_size <= WGT_BUFFER_DEPTH;
}

int Convolution::load_model(const ModelBin& mb)
{
    // printf("Convolution::load_model\n");
    if (dynamic_weight)
        return 0;

    if(int8_scale_term)
        weight_data = mb.load(u_align(weight_data_size/kernel_w/kernel_h/num_output,32)*kernel_w*kernel_h*u_align(num_output,32), 0);
    else
        weight_data = mb.load(weight_data_size, 0);

    // printf("int8_scale_term = %d, weight_data_size = %d\n",int8_scale_term,weight_data_size);
    // printf("weight_data_size_align = %d\n",u_align(weight_data_size/kernel_w/kernel_h/num_output,32)*kernel_w*kernel_h*u_align(num_output,32));

    // printf("\n");
    if (weight_data.empty())
        return -100;

    if (bias_term)
    {
        bias_data = mb.load(num_output, 1);
        if (bias_data.empty())
            return -100;
    }

    #if defined(LOG_WGT_BIAS) && !defined(NCNN_TOOLS) 

    std::string s_weight_data_name = "./log/log_weight_data_conv" + std::to_string(conv_count) + "_" + std::to_string(ifm_w) + "x" + std::to_string(ifm_h) + "x" + 
                                    std::to_string(ifm_c) + "x" + std::to_string(num_output)+ ".txt";
    std::string s_bias_data_name   = "./log/log_bias_data_conv" + std::to_string(conv_count) + "_" + std::to_string(ifm_w) + "x" + std::to_string(ifm_h) + "x" + 
                                    std::to_string(ifm_c) + "x" + std::to_string(num_output)+ ".txt";
    
    char weight_data_name[100];
    char bias_data_name[100];
    cstr(s_weight_data_name, weight_data_name);
    cstr(s_bias_data_name,   bias_data_name);

    log_mat_file<float>(bias_data  ,   bias_data_name,-1, 1, 0);
    log_mat_file<float>(weight_data, weight_data_name,-1, 1, 0);

    conv_count++;

    #endif

#if NCNN_INT8
    if (int8_scale_term != 2)
    {
        NCNN_LOGE("SESR convolution expects QAT int8_scale_term=2, got %d", int8_scale_term);
        return -1;
    }

    if (int8_scale_term)
    {
        weight_data_int8_scales = mb.load(num_output, 1);
        bottom_blob_int8_scales = mb.load(2, 1);
    }

    if (int8_scale_term > 100)
    {
        top_blob_int8_scales = mb.load(1, 1);
    }
#endif // NCNN_INT8

    return 0;
}

int Convolution::create_pipeline(const Option& opt)
{
    if (dynamic_weight)
        return 0;

    (void)(opt);

#if NCNN_INT8
    if (opt.use_int8_inference && weight_data.elemsize == (size_t)4u && int8_scale_term)
    {
        NCNN_LOGE("SESR runtime expects pre-quantized int8 convolution weights");
        return -1;
    }
#endif // NCNN_INT8

    return 0;
}

static int convolution(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data, const Mat& bias_data, int kernel_w, int kernel_h, int stride_w, int stride_h, int dilation_w, int dilation_h, int activation_type, const Mat& activation_params, const Option& opt)
{
    const int w = bottom_blob.w;
    const int inch = bottom_blob.c;

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int outch = top_blob.c;

    const int bias_term = bias_data.empty() ? 0 : 1;

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w * dilation_h - kernel_w * dilation_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }

     
    for (int p = 0; p < outch; p++)
    {
        float* outptr = top_blob.channel(p);

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                float sum = 0.f;

                if (bias_term)
                    sum = bias_data[p];

                const float* kptr = (const float*)weight_data + maxk * inch * p;

                for (int q = 0; q < inch; q++)
                {
                    const Mat m = bottom_blob.channel(q);
                    const float* sptr = m.row(i * stride_h) + j * stride_w;

                    for (int k = 0; k < maxk; k++) // 29.23
                    {
                        float val = sptr[space_ofs[k]]; // 20.72
                        float wt = kptr[k];
                        sum += val * wt; // 41.45
                    }

                    kptr += maxk;
                }

                outptr[j] = activation_ss(sum, activation_type, activation_params);
            }

            outptr += outw;
        }
    }

    return 0;
}

int Convolution::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // printf("\n\n[ log ]: 2 -> Convolution::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt)\n");
#if NCNN_INT8
    if (opt.use_int8_inference && weight_data.elemsize == (size_t)1u)
    {
        #ifdef FORWARD_ON_CPU_CONV
            return forward_int8_cpu(bottom_blob, top_blob, opt);
        #else
            return forward_int8_npu(bottom_blob, top_blob, opt);
        #endif
    }
#endif
    #if defined(FORWARD_ON_NPU_CONV) && defined(SESR_STRICT_NPU)
    NCNN_LOGE("SESR_STRICT_NPU: Convolution would use float CPU path. shape=(w=%d,h=%d,c=%d,dims=%d), weight_elemsize=%zu, use_int8=%d",
              bottom_blob.w, bottom_blob.h, bottom_blob.c, bottom_blob.dims, weight_data.elemsize, opt.use_int8_inference ? 1 : 0);
    return -100;
    #endif
    #ifdef PRINT_LAYER
    printf("[ log ]: Forward Convolution on CPU, shape = (%d, %d, %d, %d), dims = %d\n",bottom_blob.w,bottom_blob.h,bottom_blob.c,bottom_blob.d,bottom_blob.dims);
    #endif
    SESR_TRACE_OP("CPU", "Convolution", "int8=0 ifm=%dx%dx%d kernel=%dx%d stride=%dx%d outc=%d fused_act=%d",
                  bottom_blob.w, bottom_blob.h, bottom_blob.c, kernel_w, kernel_h, stride_w, stride_h, num_output, activation_type);

    if (bottom_blob.dims == 1 && kernel_w == 1 && kernel_h == 1)
    {
        NCNN_LOGE("SESR runtime does not include InnerProduct fallback");
        return -1;
    }

    Mat bottom_blob_bordered;
    make_padding(bottom_blob, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    const int w = bottom_blob_bordered.w;
    const int h = bottom_blob_bordered.h;
    const size_t elemsize = bottom_blob_bordered.elemsize;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    const int outw = (w - kernel_extent_w) / stride_w + 1;
    const int outh = (h - kernel_extent_h) / stride_h + 1;

    top_blob.create(outw, outh, num_output, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    int ret = convolution(bottom_blob_bordered, top_blob, weight_data, bias_data, kernel_w, kernel_h, stride_w, stride_h, dilation_w, dilation_h, activation_type, activation_params, opt);
    if (ret != 0)
        return ret;

    return 0;
}

int Convolution::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    (void)bottom_blobs;
    (void)top_blobs;
    (void)opt;
    NCNN_LOGE("SESR runtime does not support dynamic-weight convolution");
    return -1;
}

void Convolution::make_padding(const Mat& bottom_blob, Mat& bottom_blob_bordered, const Option& opt) const
{
    make_padding(bottom_blob, bottom_blob_bordered, kernel_w, kernel_h, opt);
}

void Convolution::make_padding(const Mat& bottom_blob, Mat& bottom_blob_bordered, int _kernel_w, int _kernel_h, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;

    const int kernel_extent_w = dilation_w * (_kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (_kernel_h - 1) + 1;

    bottom_blob_bordered = bottom_blob;
    if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0)
    {
        Option opt_b = opt;
        opt_b.blob_allocator = opt.workspace_allocator;
        copy_make_border(bottom_blob, bottom_blob_bordered, pad_top, pad_bottom, pad_left, pad_right, BORDER_CONSTANT, pad_value, opt_b);
    }
    else if (pad_left == -233 && pad_right == -233 && pad_top == -233 && pad_bottom == -233)
    {
        // tensorflow padding=SAME or onnx padding=SAME_UPPER
        int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_extent_h + (h - 1) / stride_h * stride_h - h;
        if (wpad > 0 || hpad > 0)
        {
            Option opt_b = opt;
            opt_b.blob_allocator = opt.workspace_allocator;
            copy_make_border(bottom_blob, bottom_blob_bordered, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, BORDER_CONSTANT, pad_value, opt_b);
        }
    }
    else if (pad_left == -234 && pad_right == -234 && pad_top == -234 && pad_bottom == -234)
    {
        // onnx padding=SAME_LOWER
        int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_extent_h + (h - 1) / stride_h * stride_h - h;
        if (wpad > 0 || hpad > 0)
        {
            Option opt_b = opt;
            opt_b.blob_allocator = opt.workspace_allocator;
            copy_make_border(bottom_blob, bottom_blob_bordered, hpad - hpad / 2, hpad / 2, wpad - wpad / 2, wpad / 2, BORDER_CONSTANT, pad_value, opt_b);
        }
    }
}

#if NCNN_INT8
static inline signed char float2int8(float v)
{
    int int32 = static_cast<int>(round(v));
    if (int32 > 127) return 127;
    if (int32 < -127) return -127;
    return (signed char)int32;
}

static inline unsigned char float2uint8_asym(float v, float scale, float offset)
{
    int int32 = static_cast<int>(floor((v - offset) / scale + 0.5f));
    if (int32 > 255) return 255;
    if (int32 < 0) return 0;
    return (unsigned char)int32;
}

static void quantize_to_uint8_asym(const Mat& src, Mat& dst, float scale, float offset, const Option& opt)
{
    int dims = src.dims;

    if (dims == 1)
    {
        int w = src.w;
        dst.create(w, (size_t)1u, opt.blob_allocator);
        if (dst.empty())
            return;

        const float* ptr = src;
        unsigned char* outptr = dst;
        for (int i = 0; i < w; i++)
            outptr[i] = float2uint8_asym(ptr[i], scale, offset);
    }

    if (dims == 2)
    {
        int w = src.w;
        int h = src.h;
        dst.create(w, h, (size_t)1u, opt.blob_allocator);
        if (dst.empty())
            return;

        for (int i = 0; i < h; i++)
        {
            const float* ptr = src.row(i);
            unsigned char* outptr = dst.row<unsigned char>(i);
            for (int j = 0; j < w; j++)
                outptr[j] = float2uint8_asym(ptr[j], scale, offset);
        }
    }

    if (dims == 3)
    {
        int w = src.w;
        int h = src.h;
        int channels = src.c;
        int size = w * h;
        dst.create(w, h, channels, (size_t)1u, opt.blob_allocator);
        if (dst.empty())
            return;

        for (int q = 0; q < channels; q++)
        {
            const float* ptr = src.channel(q);
            unsigned char* outptr = dst.channel(q);
            for (int i = 0; i < size; i++)
                outptr[i] = float2uint8_asym(ptr[i], scale, offset);
        }
    }
}


int Convolution::forward_int8_cpu(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    SESR_TRACE_OP("CPU", "Convolution", "int8=1 ifm=%dx%dx%d kernel=%dx%d stride=%dx%d outc=%d fused_act=%d",
                  bottom_blob.w, bottom_blob.h, bottom_blob.c, kernel_w, kernel_h, stride_w, stride_h, num_output, activation_type);
    #ifdef PRINT_LAYER
    printf("[ log ]: Forward Convolution on CPU, shape = (%d, %d, %d, %d), dims = %d\n",bottom_blob.w,bottom_blob.h,bottom_blob.c,bottom_blob.d,bottom_blob.dims);
    #endif
    assert(layout_en == 1);

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

    #ifdef SESR_VERBOSE_LOG
    printf("[ log ]: ifm_w = %d, ifm_h = %d, ifm_c = %d, kernel = %d, stride = %d\n",w,h,channels,kernel_w,stride_w);
    printf("[ log ]: pad_left = %d, pad_right = %d, pad_top = %d, pad_bottom = %d\n",pad_left, pad_right, pad_top, pad_bottom);
    #endif

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;
    if (int8_scale_term != 2)
    {
        NCNN_LOGE("SESR CPU convolution expects QAT int8_scale_term=2, got %d", int8_scale_term);
        return -100;
    }

    const float act_scale = bottom_blob_int8_scales[0];
    const float act_offset = bottom_blob_int8_scales[1];
    int act_zero_point = static_cast<int>(floor(-act_offset / act_scale + 0.5f));
    if (act_zero_point < 0) act_zero_point = 0;
    if (act_zero_point > 255) act_zero_point = 255;

    Mat bottom_blob_unbordered = bottom_blob;
    if (elemsize != 1)
    {
        Option opt_g = opt;
        opt_g.blob_allocator = opt.workspace_allocator;

        quantize_to_uint8_asym(bottom_blob, bottom_blob_unbordered, act_scale, act_offset, opt_g);
    }

    Mat bottom_blob_bordered;
    Option opt_b = opt;
    opt_b.blob_allocator = opt.workspace_allocator;
    copy_make_border(
        bottom_blob_unbordered,
        bottom_blob_bordered,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        BORDER_CONSTANT,
        (float)act_zero_point,
        opt_b);
    if (bottom_blob_bordered.empty())
        return -100;

    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;

    int outw = (w - kernel_extent_w) / stride_w + 1;
    int outh = (h - kernel_extent_h) / stride_h + 1;

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w * dilation_h - kernel_w * dilation_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }

    // int8
    bool use_int8_requantize = int8_scale_term > 100;
    size_t out_elemsize = use_int8_requantize ? 1u : 4u;

    if(use_int8_requantize){
        top_blob.create(u_align(num_output,32), 1, outw*outh, out_elemsize, opt.blob_allocator);
        memset(top_blob.data, 0, top_blob.total());
    }
    else
        top_blob.create(outw, outh, num_output, out_elemsize, opt.blob_allocator);

    if (top_blob.empty())
        return -100;
    
    Mat top_blob_t(outw, outh, num_output, out_elemsize, opt.blob_allocator);
    std::vector<int> weight_sums;
    {
        weight_sums.resize(num_output);
        for (int p = 0; p < num_output; p++)
        {
            const signed char* kptr_sum = (const signed char*)weight_data + maxk * channels * p;
            int weight_sum = 0;
            for (int kk = 0; kk < maxk * channels; kk++)
                weight_sum += (int)kptr_sum[kk];
            weight_sums[p] = weight_sum;
        }
    }

    // num_output
    for (int p = 0; p < num_output; p++)
    {
        signed char* outptr = top_blob_t.channel(p);

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                int sum = 0;

                const signed char* kptr = (const signed char*)weight_data + maxk * channels * p;

                // channels
                for (int q = 0; q < channels; q++)
                {
                    const Mat m = bottom_blob_bordered.channel(q);
                    const unsigned char* uptr = m.row<unsigned char>(i * stride_h) + j * stride_w;

                    for (int k = 0; k < maxk; k++)
                    {
                        int val = (int)uptr[space_ofs[k]];
                        int wt = kptr[k];
                        sum += val * wt;
                    }

                    kptr += maxk;
                }

                float scale_in;
                if (weight_data_int8_scales[p] == 0)
                    scale_in = 0;
                else
                    scale_in = act_scale / weight_data_int8_scales[p];

                float sumfp32 = sum * scale_in;

                if (weight_data_int8_scales[p] != 0)
                    sumfp32 += act_offset * ((float)weight_sums[p] / weight_data_int8_scales[p]);

                if (bias_term)
                    sumfp32 += bias_data[p];
                sumfp32 = activation_ss(sumfp32, activation_type, activation_params);

                if (use_int8_requantize)
                {
                    // requantize
                    float scale_out = top_blob_int8_scales[0];
                    signed char sums8 = float2int8(sumfp32 * scale_out);
                    outptr[0] = sums8;
                    outptr += 1;
                }
                else
                {
                    // dequantize
                    ((float*)outptr)[0] = sumfp32;
                    outptr += 4;
                }
                
            }
        }
    }

    if(use_int8_requantize)
        mat_nchw2nhwc(top_blob_t, top_blob);
    else
        top_blob = top_blob_t.clone();




    return 0;
}

#if defined(FORWARD_ON_NPU_CONV)
int Convolution::forward_int8_npu(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    SESR_TRACE_OP("NPU", "Convolution", "int8=1 ifm=%dx%dx%d kernel=%dx%d stride=%dx%d outc=%d fused_act=%d",
                  bottom_blob.w, bottom_blob.h, bottom_blob.c, kernel_w, kernel_h, stride_w, stride_h, num_output, activation_type);
    #ifdef PRINT_LAYER
    printf("[ log ]: Forward Convolution on NPU, shape = (%d, %d, %d, %d), dims = %d\n",bottom_blob.w,bottom_blob.h,bottom_blob.c,bottom_blob.d,bottom_blob.dims);
    #endif
    accel::hw_gemm inst;
    inst.layout_en = layout_en;
    if(ifm_w == 0 || layout_en == 1){
        // assert(0);
        inst.ifm_w = bottom_blob.w;
        inst.ifm_h = bottom_blob.h;
        inst.ifm_c = bottom_blob.c;
    }
    else{
        inst.ifm_w = ifm_w;
        inst.ifm_h = ifm_h;
        inst.ifm_c = ifm_c;
    }

    inst.kernel = kernel_w;
    inst.stride = stride_w;
    inst.padding_bottom = pad_bottom;
    inst.padding_top = pad_top;
    inst.padding_left = pad_left;
    inst.padding_right = pad_right;
    inst.ofm_c = num_output;
	conv_get_ofm_pad(inst);

    inst.oscale_en = 1;
    inst.bias_en = bias_term;
    Mat& bias_data_cast = const_cast<Mat&>(bias_data);
    inst.bias_data = (float*)bias_data_cast.data;

    if (int8_scale_term != 2)
    {
        NCNN_LOGE("SESR NPU convolution expects QAT int8_scale_term=2, got %d", int8_scale_term);
        return -100;
    }

    float act_scale = bottom_blob_int8_scales[0];
    float act_offset = bottom_blob_int8_scales[1];
    int act_zero_point = (int)floor(-act_offset / act_scale + 0.5f);
    if (act_zero_point < 0) act_zero_point = 0;
    if (act_zero_point > 255) act_zero_point = 255;
    inst.quant_zero_point = act_zero_point;
    inst.quant_scale = 1.f / act_scale;
    inst.quant_offset = act_offset;

    inst.act_op = ncnn_activation_type_to_hw(activation_type);
    if(inst.act_op != 0){
        inst.act_alpha = activation_param_or_zero(activation_params, 0);
        inst.act_beta  = activation_param_or_zero(activation_params, 1);
    }

    bool use_int8_requantize = int8_scale_term > 100;
    size_t out_elemsize = use_int8_requantize ? 1u : 4u;

    if(use_int8_requantize){
        inst.requant_en = 1;
        inst.requant_scale = top_blob_int8_scales[0];
        top_blob.create(u_align(inst.ofm_c,32), 1, inst.ofm_w*inst.ofm_h, out_elemsize, opt.blob_allocator);
    }
    else{
        top_blob.create(inst.ofm_w, inst.ofm_h, inst.ofm_c, out_elemsize, opt.blob_allocator);
    }
    
    Mat& bottom_blob_cast = const_cast<Mat&>(bottom_blob);
    Mat& weight_data_cast = const_cast<Mat&>(weight_data);
    inst.dequant_scale = (float*)weight_data_int8_scales.data;
#ifdef PRINT_SHAPE
    printf("[ log ]: ifm_w = %d, ifm_h = %d, ifm_c = %d\n",inst.ifm_w,inst.ifm_h,inst.ifm_c);
    printf("[ log ]: ofm_w = %d, ofm_h = %d, ofm_c = %d\n",inst.ofm_w,inst.ofm_h,inst.ofm_c);
    printf("[ log ]: kernel = %d, stride = %d\n",inst.kernel,inst.stride);
    printf("[ log ]: padding_left = %d, padding_right = %d, padding_top = %d, padding_bottom = %d\n",inst.padding_left,inst.padding_right,inst.padding_top,inst.padding_bottom);
    printf("[ log ]: bias_en = %d, requant_en = %d, act_op = %d, act_alpha = %f\n",inst.bias_en, inst.requant_en, inst.act_op,inst.act_alpha);
    printf("[ log ]: quant_scale = %.5f, dequant_ptr = %p, bias_ptr = %p \n",inst.quant_scale,inst.dequant_scale,inst.bias_data);
#endif


#ifdef SESR_FUSED_TAIL_STREAM_ADD_ON_NPU
    int stream_tail_armed = accel::hw_sesr_tail_add_stream_arm_if_match(inst.ofm_w, inst.ofm_h, inst.ofm_c);
    if (stream_tail_armed < 0)
    {
        return -100;
    }
    inst.no_writeback = stream_tail_armed > 0;
#endif




    inst.gemm_forward(bottom_blob_cast, weight_data_cast, top_blob);
#ifdef PRINT_BLOBS_WEIGHT
printf("quant params:\n");
printf("  act_scale       = %.8f\n", act_scale);
printf("  act_offset      = %.8f\n", act_offset);
printf("  act_zero_point  = %d\n", act_zero_point);

printf("inst params:\n");
printf("  inst.quant_zero_point = %d\n", inst.quant_zero_point);
printf("  inst.quant_scale      = %.8f\n", inst.quant_scale);
printf("  inst.quant_offset     = %.8f\n", inst.quant_offset);
printf("  inst.dequant_scale    = %p\n", (void*)inst.dequant_scale);

printf("dequant_scale / weight_data_int8_scales:\n");
for (int oc = 0; oc < num_output; oc++)
{
    printf("  oc %d: %.8f\n", oc, inst.dequant_scale[oc]);
}
printf("\n");



const float* top_blob_base = (const float*)top_blob.data;
const float* bottom_blob_base = (const float*)bottom_blob_cast.data;

printf("weight_data: elemsize=%zu total=%zu\n",
       weight_data_cast.elemsize, weight_data_cast.total());

const signed char* wptr = (const signed char*)weight_data_cast.data;
const float* bptr = bias_term ? (const float*)bias_data.data : 0;
const float* dptr = inst.dequant_scale;

int maxk = kernel_w * kernel_h;
int inch = inst.ifm_c;
int outch = inst.ofm_c;
int inch_align = (inch+31)/32*32;

for (int oc = 0; oc < outch; oc++)
{
    printf("out_channel %d:\n", oc);

    if (dptr)
        printf("  dequant_scale[%d] = %.8f\n", oc, dptr[oc]);

    if (bias_term && bptr)
        printf("  bias[%d] = %.8f\n", oc, bptr[oc]);
    else
        printf("  bias[%d] = none\n", oc);

    int oc_block = oc / 32;
    int oc_lane  = oc % 32;

    for (int ic = 0; ic < inch; ic++)
    {
        printf("  in_channel %d:\n", ic);

        for (int kh = 0; kh < kernel_h; kh++)
        {
            for (int kw = 0; kw < kernel_w; kw++)
            {
                int k = kh * kernel_w + kw;
                int j = k * inch_align + ic;

                // flip_en=1: 0..31 -> 31..0
                int j_store = (j / 32) * 32 + (31 - (j % 32));

                size_t idx =
                    (size_t)oc_block * maxk * inch_align * 32 +
                    (size_t)j_store * 32 +
                    oc_lane;

                printf("%4d ", (int)wptr[idx]);
            }
            printf("\n");
        }

        printf("\n");
    }

    printf("\n");
}
printf("bottom_blob: \n");
for (int c = 0; c < bottom_blob_cast.c; c++)
{
    printf("channel %d:\n", c);

    const float* ptr = bottom_blob_base + c * bottom_blob_cast.cstep;

    for (int y = 0; y < bottom_blob_cast.h; y++)
    {
        for (int x = 0; x < bottom_blob_cast.w; x++)
        {
            float v = ptr[y * bottom_blob_cast.w + x];
            printf("%.8f ", v);
        }
        printf("\n");
    }

    printf("\n");
}



printf("top_blob: \n");
for (int c = 0; c < top_blob.c; c++)
{
    printf("channel %d:\n", c);

    const float* ptr = top_blob_base + c * top_blob.cstep;

    for (int y = 0; y < top_blob.h; y++)
    {
        for (int x = 0; x < top_blob.w; x++)
        {
            float v = ptr[y * top_blob.w + x];
            printf("%.8f ", v);
        }
        printf("\n");
    }

    printf("\n");
}
#endif
    return 0;
}
#endif

#endif // NCNN_INT8

} // namespace ncnn


