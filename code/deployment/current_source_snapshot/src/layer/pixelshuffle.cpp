// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "pixelshuffle.h"
#include "../ncnn/platform.h"
#include "../config.h"
#include "../hw/hw_pixelshuffle.h"
#include "../utils/trace.h"

namespace ncnn {

PixelShuffle::PixelShuffle()
{
    one_blob_only = true;
    support_inplace = false;
}

int PixelShuffle::load_param(const ParamDict& pd)
{
    upscale_factor = pd.get(0, 1);
    mode = pd.get(1, 0);

    return 0;
}

int PixelShuffle::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    #ifdef PRINT_LAYER
        printf("[ log ]: Enter PixelShuffle, shape = (%d, %d, %d, %d), dims = %d\n",bottom_blob.w,bottom_blob.h,bottom_blob.c,bottom_blob.d,bottom_blob.dims);
        printf("[ log ]: PixelShuffle param: upscale_factor = %d, mode = %d\n", upscale_factor, mode);
        printf("[ log ]: PixelShuffle bottom layout: w = %d, h = %d, c = %d, d = %d, dims = %d, cstep = %zu, elemsize = %zu, elempack = %d, total = %zu, data = %p\n",
               bottom_blob.w, bottom_blob.h, bottom_blob.c, bottom_blob.d, bottom_blob.dims,
               bottom_blob.cstep, bottom_blob.elemsize, bottom_blob.elempack, bottom_blob.total(), bottom_blob.data);
    #endif
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

    int outw = w * upscale_factor;
    int outh = h * upscale_factor;
    int outc = channels / (upscale_factor * upscale_factor);

    top_blob.create(outw, outh, outc, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    #ifdef FORWARD_ON_NPU_PIXELSHUFFLE
    bool pixelshuffle_npu = bottom_blob.dims == 3 &&
                            upscale_factor == 2 &&
                            (mode == 0 || mode == 1) &&
                            channels % 4 == 0 &&
                            elemsize == 4u &&
                            bottom_blob.elempack == 1 &&
                            w % 4 == 0 &&
                            outw % 4 == 0;
    if (pixelshuffle_npu) {
        SESR_TRACE_OP("NPU", "PixelShuffle", "factor=%d mode=%d input=%dx%dx%d output=%dx%dx%d",
                      upscale_factor, mode, w, h, channels, outw, outh, outc);
        #ifdef PRINT_LAYER
            printf("[ log ]: Forward PixelShuffle on NPU, shape = (%d, %d, %d, %d), dims = %d\n", bottom_blob.w, bottom_blob.h, bottom_blob.c, bottom_blob.d, bottom_blob.dims);
        #endif
        Mat& bottom_blob_cast = const_cast<Mat&>(bottom_blob);
        accel::hw_pixelshuffle inst;
        inst.upscale_factor = upscale_factor;
        inst.mode = mode;
        inst.forward(bottom_blob_cast, top_blob);
        return 0;
    }

    #ifdef SESR_STRICT_NPU
    NCNN_LOGE("SESR_STRICT_NPU: PixelShuffle would fall back to CPU. shape=(w=%d,h=%d,c=%d,dims=%d), factor=%d, mode=%d, elemsize=%zu, elempack=%d",
              w, h, channels, bottom_blob.dims, upscale_factor, mode, elemsize, bottom_blob.elempack);
    return -100;
    #endif
    #endif

    #ifdef PRINT_LAYER
        printf("[ log ]: Forward PixelShuffle on CPU fallback\n");
    #endif
    SESR_TRACE_OP("CPU", "PixelShuffle", "factor=%d mode=%d input=%dx%dx%d output=%dx%dx%d",
                  upscale_factor, mode, w, h, channels, outw, outh, outc);

    for (int p = 0; p < outc; p++)
    {
        Mat m = top_blob.channel(p);

        for (int sh = 0; sh < upscale_factor; sh++)
        {
            for (int sw = 0; sw < upscale_factor; sw++)
            {
                int q;
                if (mode == 0)  //通道连着的插到wh中
                    q = p * upscale_factor * upscale_factor + sh * upscale_factor + sw;
                else // if (mode == 1)   //通道分散开的
                    q = (sh * upscale_factor + sw) * outc + p;

                const float* sptr = bottom_blob.channel(q);

                for (int i = 0; i < h; i++)
                {
                    float* outptr = m.row(i * upscale_factor + sh) + sw;
                    for (int j = 0; j < w; j++)
                    {
                        outptr[0] = sptr[0];

                        sptr++;
                        outptr += upscale_factor;
                    }
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
