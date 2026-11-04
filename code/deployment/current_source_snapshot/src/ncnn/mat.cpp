// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "mat.h"

#include "cpu.h"
#include "layer.h"
#include "layer_type.h"
#include "modelbin.h"
#include "paramdict.h"

#include <math.h>


namespace ncnn {

Mat Mat::clone(Allocator* _allocator) const
{
    if (empty())
        return Mat();

    Mat m;
    if (dims == 1)
        m.create(w, elemsize, elempack, _allocator);
    else if (dims == 2)
        m.create(w, h, elemsize, elempack, _allocator);
    else if (dims == 3)
        m.create(w, h, c, elemsize, elempack, _allocator);
    else if (dims == 4)
        m.create(w, h, d, c, elemsize, elempack, _allocator);

    if (total() > 0)
    {
        if (cstep == m.cstep)
            memcpy(m.data, data, total() * elemsize);
        else
        {
            // copy by channel for differnet cstep
            size_t size = (size_t)w * h * d * elemsize;
            for (int i = 0; i < c; i++)
            {
                memcpy(m.channel(i), channel(i), size);
            }
        }
    }

    return m;
}

void Mat::clone_from(const ncnn::Mat& mat, Allocator* allocator)
{
    *this = mat.clone(allocator);
}

Mat Mat::reshape(int _w, Allocator* _allocator) const
{
    if (w * h * d * c != _w)
        return Mat();

    if (dims >= 3 && cstep != (size_t)w * h * d)
    {
        Mat m;
        m.create(_w, elemsize, elempack, _allocator);

        // flatten
        for (int i = 0; i < c; i++)
        {
            const void* ptr = (unsigned char*)data + i * cstep * elemsize;
            void* mptr = (unsigned char*)m.data + (size_t)i * w * h * d * elemsize;
            memcpy(mptr, ptr, (size_t)w * h * d * elemsize);
        }

        return m;
    }

    Mat m = *this;

    m.dims = 1;
    m.w = _w;
    m.h = 1;
    m.d = 1;
    m.c = 1;

    m.cstep = _w;

    return m;
}

Mat Mat::reshape(int _w, int _h, Allocator* _allocator) const
{
    if (w * h * d * c != _w * _h)
        return Mat();

    if (dims >= 3 && cstep != (size_t)w * h * d)
    {
        Mat m;
        m.create(_w, _h, elemsize, elempack, _allocator);

        // flatten
        for (int i = 0; i < c; i++)
        {
            const void* ptr = (unsigned char*)data + i * cstep * elemsize;
            void* mptr = (unsigned char*)m.data + (size_t)i * w * h * d * elemsize;
            memcpy(mptr, ptr, (size_t)w * h * d * elemsize);
        }

        return m;
    }

    Mat m = *this;

    m.dims = 2;
    m.w = _w;
    m.h = _h;
    m.d = 1;
    m.c = 1;

    m.cstep = (size_t)_w * _h;

    return m;
}

Mat Mat::reshape(int _w, int _h, int _c, Allocator* _allocator) const
{
    if (w * h * d * c != _w * _h * _c)
        return Mat();

    if (dims < 3)
    {
        // if ((size_t)_w * _h != alignSize((size_t)_w * _h * elemsize, 16) / elemsize)
        if ((size_t)_w * _h != alignSize((size_t)_w * _h * elemsize, NCNN_MAT_ALIGN_BYTES) / elemsize)
        {
            Mat m;
            m.create(_w, _h, _c, elemsize, elempack, _allocator);

            // align channel
            for (int i = 0; i < _c; i++)
            {
                const void* ptr = (unsigned char*)data + (size_t)i * _w * _h * elemsize;
                void* mptr = (unsigned char*)m.data + i * m.cstep * m.elemsize;
                memcpy(mptr, ptr, (size_t)_w * _h * elemsize);
            }

            return m;
        }
    }
    else if (c != _c)
    {
        // flatten and then align
        Mat tmp = reshape(_w * _h * _c, _allocator);
        return tmp.reshape(_w, _h, _c, _allocator);
    }

    Mat m = *this;

    m.dims = 3;
    m.w = _w;
    m.h = _h;
    m.d = 1;
    m.c = _c;

    m.cstep = alignSize((size_t)_w * _h * elemsize, NCNN_MAT_ALIGN_BYTES) / elemsize;
    // m.cstep = alignSize((size_t)_w * _h * elemsize, 16) / elemsize;

    return m;
}

Mat Mat::reshape(int _w, int _h, int _d, int _c, Allocator* _allocator) const
{
    if (w * h * d * c != _w * _h * _d * _c)
        return Mat();

    if (dims < 3)
    {
        // if ((size_t)_w * _h * _d != alignSize((size_t)_w * _h * _d * elemsize, 16) / elemsize)
        if ((size_t)_w * _h * _d != alignSize((size_t)_w * _h * _d * elemsize, NCNN_MAT_ALIGN_BYTES) / elemsize)
        {
            Mat m;
            m.create(_w, _h, _d, _c, elemsize, elempack, _allocator);

            // align channel
            for (int i = 0; i < _c; i++)
            {
                const void* ptr = (unsigned char*)data + (size_t)i * _w * _h * _d * elemsize;
                void* mptr = (unsigned char*)m.data + i * m.cstep * m.elemsize;
                memcpy(mptr, ptr, (size_t)_w * _h * _d * elemsize);
            }

            return m;
        }
    }
    else if (c != _c)
    {
        // flatten and then align
        Mat tmp = reshape(_w * _h * _d * _c, _allocator);
        return tmp.reshape(_w, _h, _d, _c, _allocator);
    }

    Mat m = *this;

    m.dims = 4;
    m.w = _w;
    m.h = _h;
    m.d = _d;
    m.c = _c;

    // m.cstep = alignSize((size_t)_w * _h * _d * elemsize, 16) / elemsize;
    m.cstep = alignSize((size_t)_w * _h * _d * elemsize, NCNN_MAT_ALIGN_BYTES) / elemsize;

    return m;
}

void Mat::create(int _w, size_t _elemsize, Allocator* _allocator)
{
    if (dims == 1 && w == _w && elemsize == _elemsize && elempack == 1 && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = 1;
    allocator = _allocator;

    dims = 1;
    w = _w;
    h = 1;
    d = 1;
    c = 1;

    cstep = w;

    size_t totalsize = alignSize(total() * elemsize, 4);
    if (totalsize > 0)
    {
        if (allocator)
            data = allocator->fastMalloc(totalsize + (int)sizeof(*refcount));
        else
            data = fastMalloc(totalsize + (int)sizeof(*refcount));
    }

    if (data)
    {
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

void Mat::create(int _w, int _h, size_t _elemsize, Allocator* _allocator)
{
    if (dims == 2 && w == _w && h == _h && elemsize == _elemsize && elempack == 1 && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = 1;
    allocator = _allocator;

    dims = 2;
    w = _w;
    h = _h;
    d = 1;
    c = 1;

    cstep = (size_t)w * h;

    size_t totalsize = alignSize(total() * elemsize, 4);
    if (totalsize > 0)
    {
        if (allocator)
            data = allocator->fastMalloc(totalsize + (int)sizeof(*refcount));
        else
            data = fastMalloc(totalsize + (int)sizeof(*refcount));
    }

    if (data)
    {
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

void Mat::create(int _w, int _h, int _c, size_t _elemsize, Allocator* _allocator)
{
    if (dims == 3 && w == _w && h == _h && c == _c && elemsize == _elemsize && elempack == 1 && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = 1;
    allocator = _allocator;

    dims = 3;
    w = _w;
    h = _h;
    d = 1;
    c = _c;

    //return int num >= 16   16 bytes align   
    //so when dim = 3 cstep >= w*h;    total() return elem >= w*h*c
    //   when dim = 2 cstep == w*h;    total() return elem == w*h
    //   when dim = 1 cstep == w;      total() return elem == w
    // cstep = alignSize((size_t)w * h * elemsize, 16) / elemsize;
    cstep = alignSize((size_t)w * h * elemsize, NCNN_MAT_ALIGN_BYTES) / elemsize;

    size_t totalsize = alignSize(total() * elemsize, 4);
    if (totalsize > 0)
    {
        if (allocator)
            data = allocator->fastMalloc(totalsize + (int)sizeof(*refcount));
        else
            data = fastMalloc(totalsize + (int)sizeof(*refcount));
    }

    if (data)
    {
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

void Mat::create(int _w, int _h, int _d, int _c, size_t _elemsize, Allocator* _allocator)
{
    if (dims == 4 && w == _w && h == _h && d == _d && c == _c && elemsize == _elemsize && elempack == 1 && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = 1;
    allocator = _allocator;

    dims = 4;
    w = _w;
    h = _h;
    d = _d;
    c = _c;

    // cstep = alignSize((size_t)w * h * d * elemsize, 16) / elemsize;
    cstep = alignSize((size_t)w * h * d * elemsize, NCNN_MAT_ALIGN_BYTES) / elemsize;

    size_t totalsize = alignSize(total() * elemsize, 4);
    if (totalsize > 0)
    {
        if (allocator)
            data = allocator->fastMalloc(totalsize + (int)sizeof(*refcount));
        else
            data = fastMalloc(totalsize + (int)sizeof(*refcount));
    }

    if (data)
    {
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

void Mat::create(int _w, size_t _elemsize, int _elempack, Allocator* _allocator)
{
    if (dims == 1 && w == _w && elemsize == _elemsize && elempack == _elempack && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = _elempack;
    allocator = _allocator;

    dims = 1;
    w = _w;
    h = 1;
    d = 1;
    c = 1;

    cstep = w;

    size_t totalsize = alignSize(total() * elemsize, 4);
    if (totalsize > 0)
    {
        if (allocator)
            data = allocator->fastMalloc(totalsize + (int)sizeof(*refcount));
        else
            data = fastMalloc(totalsize + (int)sizeof(*refcount));
    }

    if (data)
    {
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

void Mat::create(int _w, int _h, size_t _elemsize, int _elempack, Allocator* _allocator)
{
    if (dims == 2 && w == _w && h == _h && elemsize == _elemsize && elempack == _elempack && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = _elempack;
    allocator = _allocator;

    dims = 2;
    w = _w;
    h = _h;
    d = 1;
    c = 1;

    cstep = (size_t)w * h;

    size_t totalsize = alignSize(total() * elemsize, 4);
    if (totalsize > 0)
    {
        if (allocator)
            data = allocator->fastMalloc(totalsize + (int)sizeof(*refcount));
        else
            data = fastMalloc(totalsize + (int)sizeof(*refcount));
    }

    if (data)
    {
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

void Mat::create(int _w, int _h, int _c, size_t _elemsize, int _elempack, Allocator* _allocator)
{
    if (dims == 3 && w == _w && h == _h && c == _c && elemsize == _elemsize && elempack == _elempack && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = _elempack;
    allocator = _allocator;

    dims = 3;
    w = _w;
    h = _h;
    d = 1;
    c = _c;

    cstep = alignSize((size_t)w * h * elemsize, NCNN_MAT_ALIGN_BYTES) / elemsize;
    // cstep = alignSize((size_t)w * h * elemsize, 16) / elemsize;

    size_t totalsize = alignSize(total() * elemsize, 4);
    if (totalsize > 0)
    {
        if (allocator)
            data = allocator->fastMalloc(totalsize + (int)sizeof(*refcount));
        else
            data = fastMalloc(totalsize + (int)sizeof(*refcount));
    }

    if (data)
    {
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

void Mat::create(int _w, int _h, int _d, int _c, size_t _elemsize, int _elempack, Allocator* _allocator)
{
    if (dims == 4 && w == _w && h == _h && d == _d && c == _c && elemsize == _elemsize && elempack == _elempack && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = _elempack;
    allocator = _allocator;

    dims = 4;
    w = _w;
    h = _h;
    d = _d;
    c = _c;

    cstep = alignSize((size_t)w * h * d * elemsize, NCNN_MAT_ALIGN_BYTES) / elemsize;
    // cstep = alignSize((size_t)w * h * d * elemsize, 16) / elemsize;

    size_t totalsize = alignSize(total() * elemsize, 4);
    if (totalsize > 0)
    {
        if (allocator)
            data = allocator->fastMalloc(totalsize + (int)sizeof(*refcount));
        else
            data = fastMalloc(totalsize + (int)sizeof(*refcount));
    }

    if (data)
    {
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

void Mat::create_like(const Mat& m, Allocator* _allocator)
{
    int _dims = m.dims;
    if (_dims == 1)
        create(m.w, m.elemsize, m.elempack, _allocator);
    if (_dims == 2)
        create(m.w, m.h, m.elemsize, m.elempack, _allocator);
    if (_dims == 3)
        create(m.w, m.h, m.c, m.elemsize, m.elempack, _allocator);
    if (_dims == 4)
        create(m.w, m.h, m.d, m.c, m.elemsize, m.elempack, _allocator);
}

void Mat::substract_mean_normalize(const float* mean_vals, const float* norm_vals)
{
    if (!mean_vals && !norm_vals)
        return;

    if (elemsize != 4u)
    {
        NCNN_LOGE("substract_mean_normalize expects fp32 Mat");
        return;
    }

    const int channels = dims >= 3 ? c : 1;
    const size_t plane_size = dims == 1 ? (size_t)w :
                              dims == 2 ? (size_t)w * h :
                              dims == 3 ? (size_t)w * h :
                                           (size_t)w * h * d;

    for (int q = 0; q < channels; q++)
    {
        const float mean = mean_vals ? mean_vals[q] : 0.f;
        const float norm = norm_vals ? norm_vals[q] : 1.f;
        float* ptr = dims >= 3 ? (float*)channel(q) : (float*)data;

        for (size_t i = 0; i < plane_size; i++)
        {
            ptr[i] = (ptr[i] - mean) * norm;
        }
    }
}

Mat Mat::from_float16(const unsigned short* data, int size)
{
    Mat m(size);
    if (m.empty())
        return m;

    float* ptr = m; //.data;

    int remain = size;

    for (; remain > 0; remain--)
    {
        *ptr = float16_to_float32(*data);

        data++;
        ptr++;
    }

    return m;
}


unsigned short float32_to_float16(float value)
{
    // 1 : 8 : 23
    union
    {
        unsigned int u;
        float f;
    } tmp;

    tmp.f = value;

    // 1 : 8 : 23
    unsigned short sign = (tmp.u & 0x80000000) >> 31;
    unsigned short exponent = (tmp.u & 0x7F800000) >> 23;
    unsigned int significand = tmp.u & 0x7FFFFF;

    //     NCNN_LOGE("%d %d %d", sign, exponent, significand);

    // 1 : 5 : 10
    unsigned short fp16;
    if (exponent == 0)
    {
        // zero or denormal, always underflow
        fp16 = (sign << 15) | (0x00 << 10) | 0x00;
    }
    else if (exponent == 0xFF)
    {
        // infinity or NaN
        fp16 = (sign << 15) | (0x1F << 10) | (significand ? 0x200 : 0x00);
    }
    else
    {
        // normalized
        short newexp = exponent + (-127 + 15);
        if (newexp >= 31)
        {
            // overflow, return infinity
            fp16 = (sign << 15) | (0x1F << 10) | 0x00;
        }
        else if (newexp <= 0)
        {
            // Some normal fp32 cannot be expressed as normal fp16
            fp16 = (sign << 15) | (0x00 << 10) | 0x00;
        }
        else
        {
            // normal fp16
            fp16 = (sign << 15) | (newexp << 10) | (significand >> 13);
        }
    }

    return fp16;
}

float float16_to_float32(unsigned short value)
{
    // 1 : 5 : 10
    unsigned short sign = (value & 0x8000) >> 15;
    unsigned short exponent = (value & 0x7c00) >> 10;
    unsigned short significand = value & 0x03FF;

    //     NCNN_LOGE("%d %d %d", sign, exponent, significand);

    // 1 : 8 : 23
    union
    {
        unsigned int u;
        float f;
    } tmp;
    if (exponent == 0)
    {
        if (significand == 0)
        {
            // zero
            tmp.u = (sign << 31);
        }
        else
        {
            // denormal
            exponent = 0;
            // find non-zero bit
            while ((significand & 0x200) == 0)
            {
                significand <<= 1;
                exponent++;
            }
            significand <<= 1;
            significand &= 0x3FF;
            tmp.u = (sign << 31) | ((-exponent + (-15 + 127)) << 23) | (significand << 13);
        }
    }
    else if (exponent == 0x1F)
    {
        // infinity or NaN
        tmp.u = (sign << 31) | (0xFF << 23) | (significand << 13);
    }
    else
    {
        // normalized
        tmp.u = (sign << 31) | ((exponent + (-15 + 127)) << 23) | (significand << 13);
    }

    return tmp.f;
}

void copy_make_border(const Mat& src, Mat& dst, int top, int bottom, int left, int right, int type, float v, const Option& opt)
{
    Layer* padding = create_layer(LayerType::Padding);

    ParamDict pd;
    pd.set(0, top);
    pd.set(1, bottom);
    pd.set(2, left);
    pd.set(3, right);
    pd.set(4, type);
    pd.set(5, v);

    padding->load_param(pd);

    padding->create_pipeline(opt);

    padding->forward(src, dst, opt);

    padding->destroy_pipeline(opt);

    delete padding;
}

void copy_make_border_3d(const Mat& src, Mat& dst, int top, int bottom, int left, int right, int front, int behind, int type, float v, const Option& opt)
{
    Layer* padding = create_layer(LayerType::Padding);

    ParamDict pd;
    pd.set(0, top);
    pd.set(1, bottom);
    pd.set(2, left);
    pd.set(3, right);
    pd.set(4, type);
    pd.set(5, v);
    pd.set(7, front);
    pd.set(8, behind);

    padding->load_param(pd);

    padding->create_pipeline(opt);

    padding->forward(src, dst, opt);

    padding->destroy_pipeline(opt);

    delete padding;
}

void copy_cut_border(const Mat& src, Mat& dst, int top, int bottom, int left, int right, const Option& opt)
{
    (void)opt;
    if (left + right > src.w || top + bottom > src.h)
    {
        NCNN_LOGE("copy_cut_border parameter error, top: %d, bottom: %d, left: %d, right: %d, src.w: %d, src.h: %d", top, bottom, left, right, src.w, src.h);
        return;
    }

    const int outw = src.w - left - right;
    const int outh = src.h - top - bottom;
    if (src.dims == 2)
    {
        dst.create(outw, outh, src.elemsize, src.elempack, opt.blob_allocator);
        for (int y = 0; y < outh; y++)
        {
            memcpy((unsigned char*)dst.row(y), (const unsigned char*)src.row(y + top) + (size_t)left * src.elemsize, (size_t)outw * src.elemsize);
        }
        return;
    }

    if (src.dims == 3)
    {
        dst.create(outw, outh, src.c, src.elemsize, src.elempack, opt.blob_allocator);
        for (int q = 0; q < src.c; q++)
        {
            Mat dstm = dst.channel(q);
            const Mat srcm = src.channel(q);
            for (int y = 0; y < outh; y++)
            {
                memcpy((unsigned char*)dstm.row(y), (const unsigned char*)srcm.row(y + top) + (size_t)left * src.elemsize, (size_t)outw * src.elemsize);
            }
        }
        return;
    }

    NCNN_LOGE("copy_cut_border supports only 2D/3D tensors in the SESR runtime");
}

void copy_cut_border_3d(const Mat& src, Mat& dst, int top, int bottom, int left, int right, int front, int behind, const Option& opt)
{
    (void)src;
    (void)dst;
    (void)top;
    (void)bottom;
    (void)left;
    (void)right;
    (void)front;
    (void)behind;
    (void)opt;
    NCNN_LOGE("copy_cut_border_3d is not supported in the SESR runtime");
}

void resize_nearest(const Mat& src, Mat& dst, int w, int h, const Option& opt)
{
    (void)src;
    (void)dst;
    (void)w;
    (void)h;
    (void)opt;
    NCNN_LOGE("resize_nearest is not supported in the SESR runtime");
}

void resize_bilinear(const Mat& src, Mat& dst, int w, int h, const Option& opt)
{
    (void)src;
    (void)dst;
    (void)w;
    (void)h;
    (void)opt;
    NCNN_LOGE("resize_bilinear is not supported in the SESR runtime");
}

void resize_bicubic(const Mat& src, Mat& dst, int w, int h, const Option& opt)
{
    (void)src;
    (void)dst;
    (void)w;
    (void)h;
    (void)opt;
    NCNN_LOGE("resize_bicubic is not supported in the SESR runtime");
}

void convert_packing(const Mat& src, Mat& dst, int _elempack, const Option& opt)
{
    if (src.elempack == _elempack)
    {
        dst = src.clone(opt.blob_allocator);
        return;
    }

    NCNN_LOGE("convert_packing is not supported in the SESR runtime");
}

void flatten(const Mat& src, Mat& dst, const Option& opt)
{
    dst = src.reshape((int)src.total(), opt.blob_allocator);
}

static void create_like_shape(Mat& dst, const Mat& src, size_t elemsize, const Option& opt)
{
    if (src.dims == 1)
        dst.create(src.w, elemsize, opt.blob_allocator);
    else if (src.dims == 2)
        dst.create(src.w, src.h, elemsize, opt.blob_allocator);
    else if (src.dims == 3)
        dst.create(src.w, src.h, src.c, elemsize, opt.blob_allocator);
    else if (src.dims == 4)
        dst.create(src.w, src.h, src.d, src.c, elemsize, opt.blob_allocator);
}

void cast_float32_to_float16(const Mat& src, Mat& dst, const Option& opt)
{
    create_like_shape(dst, src, (size_t)2u, opt);
    if (dst.empty())
        return;
    const float* sp = src;
    unsigned short* dp = dst;
    for (size_t i = 0; i < src.total(); i++)
        dp[i] = float32_to_float16(sp[i]);
}

void cast_float16_to_float32(const Mat& src, Mat& dst, const Option& opt)
{
    create_like_shape(dst, src, (size_t)4u, opt);
    if (dst.empty())
        return;
    const unsigned short* sp = src;
    float* dp = dst;
    for (size_t i = 0; i < src.total(); i++)
        dp[i] = float16_to_float32(sp[i]);
}

void cast_int8_to_float32(const Mat& src, Mat& dst, const Option& opt)
{
    create_like_shape(dst, src, (size_t)4u, opt);
    if (dst.empty())
        return;
    const signed char* sp = src;
    float* dp = dst;
    for (size_t i = 0; i < src.total(); i++)
        dp[i] = (float)sp[i];
}

void cast_float32_to_bfloat16(const Mat& src, Mat& dst, const Option& opt)
{
    create_like_shape(dst, src, (size_t)2u, opt);
    if (dst.empty())
        return;
    const float* sp = src;
    unsigned short* dp = dst;
    for (size_t i = 0; i < src.total(); i++)
        dp[i] = float32_to_bfloat16(sp[i]);
}

void cast_bfloat16_to_float32(const Mat& src, Mat& dst, const Option& opt)
{
    create_like_shape(dst, src, (size_t)4u, opt);
    if (dst.empty())
        return;
    const unsigned short* sp = src;
    float* dp = dst;
    for (size_t i = 0; i < src.total(); i++)
        dp[i] = bfloat16_to_float32(sp[i]);
}

void quantize_to_int8(const Mat& src, Mat& dst, const Mat& scale_data, const Option& opt)
{
    (void)src;
    (void)dst;
    (void)scale_data;
    (void)opt;
    NCNN_LOGE("quantize_to_int8 is not supported in the SESR runtime");
}

void dequantize_from_int32(const Mat& src, Mat& dst, const Mat& scale_data, const Mat& bias_data, const Option& opt)
{
    (void)src;
    (void)dst;
    (void)scale_data;
    (void)bias_data;
    (void)opt;
    NCNN_LOGE("dequantize_from_int32 is not supported in the SESR runtime");
}

void requantize_from_int32_to_int8(const Mat& src, Mat& dst, const Mat& scale_in_data, const Mat& scale_out_data, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt)
{
    (void)src;
    (void)dst;
    (void)scale_in_data;
    (void)scale_out_data;
    (void)bias_data;
    (void)activation_type;
    (void)activation_params;
    (void)opt;
    NCNN_LOGE("requantize_from_int32_to_int8 is not supported in the SESR runtime");
}

} // namespace ncnn
