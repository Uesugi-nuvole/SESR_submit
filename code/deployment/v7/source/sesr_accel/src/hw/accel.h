#ifndef __ACCEL_H_
#define __ACCEL_H_
#include "../net/timer.h"
#include <stdio.h>
#include <stdlib.h>
#include "pl_intr.h"
#include "accel_params.h"
#include "xil_cache.h"
#include "../ncnn/mat.h"

#ifndef u_align
#define u_align(x,n) (((x)+(n)-1)&-((n)))
#endif
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
    unsigned long polls=0;
    while(!gemm_task_done) {
        if(++polls>=100000000UL) {
            printf("[FAIL] GEMM interrupt wait exceeded poll budget\n");fflush(stdout);exit(6);
        }
    }
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
