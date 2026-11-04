#pragma once
#include <stdint.h>
#ifndef COMPLETION_TIMER_HOST_TEST
#include "xil_io.h"
#include "../hw/accel_params.h"
#endif
// Fixed retry budget: timing failure must not prevent output readback.
static bool completion_cycle_snapshot(uint64_t& value) {
    for(unsigned retry=0;retry<8;retry++) {
        uint32_t hi1=Xil_In32(ACCEL_REG_PERF_CYCLE_HI_ADDR);
        uint32_t lo=Xil_In32(ACCEL_REG_PERF_CYCLE_LO_ADDR);
        uint32_t hi2=Xil_In32(ACCEL_REG_PERF_CYCLE_HI_ADDR);
        if(hi1==hi2) {value=(uint64_t(hi1)<<32)|lo;return true;}
    }
    return false;
}
