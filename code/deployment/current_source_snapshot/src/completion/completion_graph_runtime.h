#ifndef COMPLETION_GRAPH_RUNTIME_H
#define COMPLETION_GRAPH_RUNTIME_H
#include "completion_cpu_ops.h"
#ifndef COMPLETION_GRAPH_CPU
#include "../hw/hw_gemm.h"
#include "../hw/pl_intr.h"
#include "xil_cache.h"
#endif
#include "completion_assets.h"
#include "completion_static_graph.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <cfenv>
#include <cstdint>
namespace cg {
struct Tensor {int c,h,w;std::vector<float> v;Tensor():c(0),h(0),w(0){} };
static void read(int i,const char* field,void* p,size_t bytes) {
    char name[80];snprintf(name,sizeof(name),"%03d_%s.bin",i,field);
    for(size_t k=0;k<sizeof(completion_assets)/sizeof(completion_assets[0]);++k) {
        const CompletionAsset& asset=completion_assets[k];
        if(strcmp(name,asset.name)==0 && bytes==asset.bytes &&
           completion_copy_asset(asset.offset,p,bytes,strcmp(field,"weight")==0)) return;
    }
    printf("[FAIL] asset %s bytes=%lu\n",name,(unsigned long)bytes);exit(2);
}
static std::vector<float> load(int i,const char* field,size_t n) {
    std::vector<float> v(n);read(i,field,v.data(),n*4);return v;
}
static float quant(float x,float scale,int low,int high,bool dequant) {
    float v=x/scale;v=v<low?float(low):(v>high?float(high):v);
    float base=std::floor(v),fraction=v-base;
    float q=(fraction>0.5f || (fraction==0.5f && std::fmod(base,2.f)!=0.f))?base+1.f:base;
    return dequant?((q-v)+v)*scale:q;
}
static void convolution(int id,const Tensor& x,Tensor& y,int stride,float scale) {
    auto bias=load(id,"bias",y.c);
#ifdef COMPLETION_GRAPH_CPU
#ifdef COMPLETION_GRAPH_INTEGER
    // Independent integer-MAC reference. This does not replace PyTorch refs.
    int ia=(x.c+31)/32*32,oa=(y.c+31)/32*32;
    std::vector<int8_t> packed(ia*oa*9);
    read(id,"weight",packed.data(),packed.size());
    auto scales=load(id,"scales",y.c);
    std::vector<int> q(x.v.size());
    for(size_t j=0;j<q.size();j++)q[j]=int(quant(x.v[j],scale,0,255,false));
    for(int o=0;o<y.c;o++)for(int r=0;r<y.h;r++)for(int c=0;c<y.w;c++) {
        int64_t acc=0;
        for(int i=0;i<x.c;i++)for(int kr=0;kr<3;kr++)for(int kc=0;kc<3;kc++) {
            int rr=r*stride+kr-1,cc=c*stride+kc-1;
            if(rr>=0 && rr<x.h && cc>=0 && cc<x.w) {
                int lane=(i/32)*32+31-i%32;
                size_t wi=(o/32)*(9*ia*32)+((kr*3+kc)*ia+lane)*32+o%32;
                acc+=int64_t(q[(i*x.h+rr)*x.w+cc])*int(packed[wi]);
            }
        }
        float scaled=float(acc)*scales[o];
        y.v[(o*y.h+r)*y.w+c]=scaled+bias[o];
    }
#else
    auto w=load(id,"weight_float",y.c*x.c*9);
    std::vector<float> q(x.v.size());for(size_t j=0;j<q.size();j++)q[j]=quant(x.v[j],scale,0,255,true);
    for(int o=0;o<y.c;o++)for(int r=0;r<y.h;r++)for(int c=0;c<y.w;c++) {
        float sum=0.f;
        for(int i=0;i<x.c;i++)for(int kr=0;kr<3;kr++)for(int kc=0;kc<3;kc++) {
            int rr=r*stride+kr-1,cc=c*stride+kc-1;
            if(rr>=0 && rr<x.h && cc>=0 && cc<x.w)sum+=q[(i*x.h+rr)*x.w+cc]*w[((o*x.c+i)*3+kr)*3+kc];
        }
        y.v[(o*y.h+r)*y.w+c]=sum+bias[o];
    }
#endif
#else
    int ac=(y.c+31)/32*32,wb=ac*((x.c+31)/32*32)*9;
    ncnn::Mat in(x.w,x.h,x.c),out(y.w,y.h,y.c),weight(wb,(size_t)1u),scales(ac),biases(ac);
    if(!in.data || !out.data || !weight.data || !scales.data || !biases.data) {
        printf("[FAIL] NPU allocation node=%d\n",id);fflush(stdout);exit(7);
    }
    memset(in.data,0,in.cstep*x.c*4);memset(scales.data,0,ac*4);memset(biases.data,0,ac*4);
    for(int c=0;c<x.c;c++)for(int p=0;p<x.h*x.w;p++)((float*)in.data)[c*in.cstep+p]=quant(x.v[c*x.h*x.w+p],scale,0,255,false);
    for(size_t p=0;p<out.cstep*y.c;p++)((float*)out.data)[p]=NAN;
    read(id,"weight",weight.data,wb);read(id,"scales",scales.data,y.c*4);memcpy(biases.data,bias.data(),y.c*4);
#ifdef COMPLETION_CPU_POSTSCALE
    std::vector<float> saved_scales(y.c),saved_bias=bias;
    memcpy(saved_scales.data(),scales.data,y.c*4);
    // All supported ordinary convolutions have at most 32 input channels:
    // 32*9*255*128 < 2^24, hence int accumulator -> float32 is exact.
    if(int64_t(x.c)*9*255*128 >= (int64_t(1)<<24)) {
        fprintf(stderr,"[ERR] integer accumulator exact-float bound exceeded\n");exit(5);
    }
    for(int j=0;j<ac;j++){((float*)scales.data)[j]=1.f;((float*)biases.data)[j]=0.f;}
#endif
    accel::hw_gemm op;op.ifm_w=x.w;op.ifm_h=x.h;op.ifm_c=x.c;op.ofm_w=y.w;op.ofm_h=y.h;op.ofm_c=y.c;
    op.stride=stride;op.quant_scale=1.f;op.quant_offset=0.f;op.quant_zero_point=0;op.act_prop=0;
    op.dequant_scale=(float*)scales.data;op.bias_data=(float*)biases.data;
    // Use the original board driver's tiled path for 128x128 inputs.
    Xil_DCacheFlush();
    gemm_task_done=0;
    op.gemm_forward(in,weight,out);
    Xil_DCacheInvalidateRange((INTPTR)out.data,out.cstep*y.c*4);
    for(int c=0;c<y.c;c++)memcpy(y.v.data()+c*y.h*y.w,(float*)out.data+c*out.cstep,y.h*y.w*4);
#ifdef COMPLETION_CPU_POSTSCALE
    for(int c=0;c<y.c;c++)for(int p=0;p<y.h*y.w;p++) {
        size_t j=c*y.h*y.w+p;float accumulator=y.v[j];
        if(!std::isfinite(accumulator) || accumulator!=std::nearbyint(accumulator)) {
            fprintf(stderr,"[ERR] non-integer NPU MAC output node=%d index=%zu\n",id,j);exit(5);
        }
        float scaled=accumulator*saved_scales[c];y.v[j]=scaled+saved_bias[c];
    }
#endif
#endif
}
static int run() {
    // MicroBlaze newlib has no FE_TONEAREST; quantizer rounds ties explicitly.
#ifdef COMPLETION_GRAPH_CPU
#ifdef COMPLETION_GRAPH_INTEGER
    const char* mode="integer";
#else
    const char* mode="cpu";
#endif
#else
#ifdef COMPLETION_CPU_POSTSCALE
    const char* mode="hybrid_postscale";
#else
    const char* mode="hybrid";
#endif
#endif
    const int count=completion_node_count,final=completion_final_node;
    static std::vector<Tensor> tensors;
    tensors.clear();tensors.resize(count);
    std::vector<int> last(count);
    for(int i=0;i<count;i++)last[i]=i;
    for(int i=0;i<count;i++) {
        if(completion_nodes[i].a>=0)last[completion_nodes[i].a]=i;
        if(completion_nodes[i].b>=0)last[completion_nodes[i].b]=i;
    }
    last[final]=count;
    for(int node=0;node<count;node++) {
        const CompletionNode& n=completion_nodes[node];
        int id=n.id,kind=n.kind,a=n.a,b=n.b,c=n.c,h=n.h,w=n.w,p=n.p;float scale=n.scale;
        if(a>=node || b>=node || (kind!=0 && a<0))return 2;
        Tensor& y=tensors[node];y.c=c;y.h=h;y.w=w;y.v.resize(c*h*w);
        const Tensor& x=a>=0?tensors[a]:y;
        printf("[NODE] mode=%s id=%d kind=%d shape=%dx%dx%d\n",mode,id,kind,c,h,w);fflush(stdout);
        if(kind==0)read(id,"input",y.v.data(),y.v.size()*4);
        else if(kind==1)convolution(id,x,y,p,scale);
        else if(kind==2) {
            auto weight=load(id,"weight_float",x.c*c*9);auto bias=load(id,"bias",c);
            std::vector<float> q(x.v.size());for(size_t j=0;j<q.size();j++)q[j]=quant(x.v[j],scale,0,255,true);
            completion::transpose3x3_s2(q.data(),weight.data(),bias.data(),y.v.data(),x.c,x.h,x.w,c);
        } else if(kind==3)for(size_t j=0;j<y.v.size();j++)y.v[j]=x.v[j]>0?x.v[j]:0.f;
        else if(kind==4)for(size_t j=0;j<y.v.size();j++)y.v[j]=quant(x.v[j],scale,-128,127,true);
        else if(kind==5 || kind==6)for(size_t j=0;j<y.v.size();j++) {
            float rhs=b>=0?tensors[b].v[j]:scale;y.v[j]=kind==5?x.v[j]+rhs:x.v[j]/rhs;
        } else if(kind==7) {
            if(b<0 || x.v.size()+tensors[b].v.size()!=y.v.size())return 2;
            memcpy(y.v.data(),x.v.data(),x.v.size()*4);memcpy(y.v.data()+x.v.size(),tensors[b].v.data(),tensors[b].v.size()*4);
        } else if(kind==8) {
            for(int ch=0;ch<c;ch++)for(int r=0;r<h;r++)for(int col=0;col<w;col++) {
                float sum=0;for(int kr=0;kr<p;kr++)for(int kc=0;kc<p;kc++)sum+=x.v[(ch*x.h+r*p+kr)*x.w+col*p+kc];
                y.v[(ch*h+r)*w+col]=sum/float(p*p);
            }
        } else if(kind==9)completion::bilinear_align_corners(x.v.data(),y.v.data(),c,x.h,x.w,h,w);
        else return 2;
        for(size_t j=0;j<y.v.size();j++)if(!std::isfinite(y.v[j])) {
            printf("[FAIL] nonfinite node=%d\n",id);return 4;
        }
        for(int j=0;j<=node;j++)if(last[j]==node)std::vector<float>().swap(tensors[j].v);
    }
    Tensor& result=tensors[final];
#ifndef COMPLETION_GRAPH_CPU
    Xil_DCacheFlushRange((INTPTR)result.v.data(),result.v.size()*4);
#endif
#ifdef COMPLETION_OUTPUT_HOOK
    COMPLETION_OUTPUT_HOOK(result.v.data(),result.v.size());
#endif
    printf("[DONE] node=%d shape=%dx%dx%d address=%p bytes=%lu\n",final,result.c,result.h,result.w,
        (void*)result.v.data(),(unsigned long)(result.v.size()*4));
    return 0;
}
}
static int completion_board_run(){return cg::run();}
#endif
