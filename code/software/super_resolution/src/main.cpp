#include <stdio.h>
#include "xil_printf.h"
#include "xil_io.h"
 #include "sleep.h"
#include "net/cnn_demo.h"
#include "hw/pl_intr.h"
#include "hw/dri_sd.h"
#include "hw/accel_params.h"
#include "xil_cache.h"
#include "config.h"

int main(){
	init_intr_sys();
	demo::sesr_inference();
	return 0;
}

