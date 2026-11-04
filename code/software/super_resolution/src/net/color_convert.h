#ifndef SESR_COLOR_CONVERT_H
#define SESR_COLOR_CONVERT_H

#include "../ncnn/simpleocv.h"

namespace demo {

struct YCbCrImage
{
    cv::Mat y;
    cv::Mat cb;
    cv::Mat cr;
    int image_w;
    int image_h;

    YCbCrImage()
        : image_w(0), image_h(0)
    {
    }

    bool empty() const
    {
        return cb.empty() || cr.empty() || image_w <= 0 || image_h <= 0;
    }

    int width() const
    {
        return image_w;
    }

    int height() const
    {
        return image_h;
    }
};

void rgb_to_ycbcr_pixel(float r, float g, float b, float& y, float& cb, float& cr);
void ycbcr_to_rgb_pixel(float y, float cb, float cr, float& r, float& g, float& b);

int bgr_to_ycbcr(const cv::Mat& bgr, YCbCrImage& ycbcr);
int bgr_to_ycbcr_and_y_input(const cv::Mat& bgr, YCbCrImage& ycbcr, ncnn::Mat& y_input);
int bgr_to_ycbcr_and_y_input_npu(const cv::Mat& bgr, YCbCrImage& ycbcr, ncnn::Mat& y_input, bool save_y=0);
int ycbcr_to_bgr(const YCbCrImage& ycbcr, cv::Mat& bgr);
int compose_bgr_from_y_and_lr_chroma(const cv::Mat& y_sr, const cv::Mat& cb_lr, const cv::Mat& cr_lr, int scale, cv::Mat& bgr);
int compose_bgr_from_y_float_and_lr_chroma(const ncnn::Mat& y_sr, const cv::Mat& cb_lr, const cv::Mat& cr_lr, int scale, cv::Mat& bgr);
int compose_bgr_from_y_float_and_lr_chroma_npu(const ncnn::Mat& y_sr, const cv::Mat& cb_lr, const cv::Mat& cr_lr, int scale, cv::Mat& bgr);

} // namespace demo

#endif // SESR_COLOR_CONVERT_H
