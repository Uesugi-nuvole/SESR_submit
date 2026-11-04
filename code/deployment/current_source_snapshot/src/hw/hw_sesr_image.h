#ifndef __HW_SESR_IMAGE_H_
#define __HW_SESR_IMAGE_H_

#include "../ncnn/mat.h"
#include "../ncnn/simpleocv.h"

namespace accel {

int hw_sesr_preprocess_bgr_to_ycbcr_yinput(const cv::Mat& bgr, cv::Mat& y, cv::Mat& cb, cv::Mat& cr,
                                           ncnn::Mat& y_input, bool save_y = true);
int hw_sesr_postprocess_ycbcr_to_bgr(const ncnn::Mat& y_sr, const cv::Mat& cb_lr, const cv::Mat& cr_lr, int scale, cv::Mat& bgr);
int hw_sesr_tail_fused_to_bgr(const ncnn::Mat& tail_blob, const cv::Mat& cb_lr, const cv::Mat& cr_lr, int scale, cv::Mat& bgr);
int hw_sesr_tail_add_fused_to_bgr(const ncnn::Mat& tail_blob, const ncnn::Mat& skip_y,
                                  const cv::Mat& cb_lr, const cv::Mat& cr_lr, int scale, cv::Mat& bgr);
int hw_sesr_tail_add_stream_begin(int width, int height, const ncnn::Mat& skip_y,
                                  const cv::Mat& cb_lr, const cv::Mat& cr_lr, int scale, cv::Mat& bgr);
void hw_sesr_tail_add_stream_wait();
void hw_sesr_tail_add_stream_prepare(int width, int height, const ncnn::Mat& skip_y,
                                     const cv::Mat& cb_lr, const cv::Mat& cr_lr, int scale, cv::Mat& bgr);
int hw_sesr_tail_add_stream_arm_if_match(int ofm_w, int ofm_h, int ofm_c);
void hw_sesr_tail_add_stream_clear();
int hw_sesr_split_ack(const ncnn::Mat& bottom_blob, size_t outputs);

} // namespace accel

#endif
