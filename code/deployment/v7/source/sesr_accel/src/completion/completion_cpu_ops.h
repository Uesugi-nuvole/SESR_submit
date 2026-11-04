#ifndef COMPLETION_CPU_OPS_H
#define COMPLETION_CPU_OPS_H
#include <cstddef>
namespace completion {
// Contiguous CHW, batch=1. Weights are already dequantized IOHW from the
// original model (per-input-channel weight scales must not be moved to output).
inline void transpose3x3_s2(const float* x,const float* w,const float* b,
                           float* y,int ic,int ih,int iw,int oc) {
    const int oh=ih*2,ow=iw*2;
    for(int o=0;o<oc;o++)for(int r=0;r<oh;r++)for(int c=0;c<ow;c++) {
        float acc=0.f;
        for(int i=0;i<ic;i++)for(int kr=0;kr<3;kr++)for(int kc=0;kc<3;kc++) {
            int rr=r+1-kr,cc=c+1-kc;
            if(rr<0 || cc<0 || rr%2 || cc%2)continue;
            rr/=2;cc/=2;if(rr>=ih || cc>=iw)continue;
            acc+=x[(i*ih+rr)*iw+cc]*w[((i*oc+o)*3+kr)*3+kc];
        }
        y[(o*oh+r)*ow+c]=acc+b[o];
    }
}
inline void bilinear_align_corners(const float* x,float* y,int ch,int ih,int iw,int oh,int ow) {
    const float sy=oh>1?float(ih-1)/float(oh-1):0.f;
    const float sx=ow>1?float(iw-1)/float(ow-1):0.f;
    for(int c=0;c<ch;c++)for(int r=0;r<oh;r++)for(int col=0;col<ow;col++) {
        float fy=sy*r,fx=sx*col;int y0=int(fy),x0=int(fx);
        int y1=y0<ih-1?y0+1:y0,x1=x0<iw-1?x0+1:x0;
        float dy=fy-y0,dx=fx-x0;
        const float* p=x+c*ih*iw;
        y[(c*oh+r)*ow+col]=(1.f-dy)*((1.f-dx)*p[y0*iw+x0]+dx*p[y0*iw+x1])
                              +dy*((1.f-dx)*p[y1*iw+x0]+dx*p[y1*iw+x1]);
    }
}
// Original preprocessing: avg_pool(depth)/(avg_pool(mask)+0.0001).
inline void normalized_pool(const float* depth,const float* mask,float* out,int h,int w,int k) {
    for(int r=0;r<h/k;r++)for(int c=0;c<w/k;c++) {
        float ds=0.f,ms=0.f;
        for(int kr=0;kr<k;kr++)for(int kc=0;kc<k;kc++) {
            int p=(r*k+kr)*w+c*k+kc;ds+=depth[p];ms+=mask[p];
        }
        out[r*(w/k)+c]=(ds/float(k*k))/(ms/float(k*k)+0.0001f);
    }
}
}
#endif
