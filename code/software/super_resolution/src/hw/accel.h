#ifndef __ACCEL_H_
#define __ACCEL_H_
#include "../net/timer.h"
#include "pl_intr.h"
#include "accel_params.h"
#include "xil_cache.h"
#include "../ncnn/mat.h"

#define u_align(x,n)  ((x+n-1) & -n)
static  u32 t1,t2;
static inline void wait_alu_mat_done(){
	t1=TOTAL_CYCLES;
    while(!alu_mat_task_done);
    t2=TOTAL_CYCLES;
    alu_mat_task_done = 0;
    npu_cycles+=t2-t1;

}

static inline void wait_pool_done(){
	t1=TOTAL_CYCLES;
    while(!pool_task_done);
    t2=TOTAL_CYCLES;
    pool_task_done = 0;
    npu_cycles+=t2-t1;
}

static inline void wait_gemm_done(){
	t1=TOTAL_CYCLES;
    while(!gemm_task_done);
    t2=TOTAL_CYCLES;
    gemm_task_done = 0;
    npu_cycles+=t2-t1;
}

static inline void wait_pixelshuffle_done(){
	t1=TOTAL_CYCLES;
    while(!pixelshuffle_task_done);
    t2=TOTAL_CYCLES;
    pixelshuffle_task_done = 0;
    npu_cycles+=t2-t1;
}

static inline void wait_image_done(){
	t1=TOTAL_CYCLES;
    while(!image_task_done);
    t2=TOTAL_CYCLES;
    image_task_done = 0;
    npu_cycles+=t2-t1;
}
#endif
