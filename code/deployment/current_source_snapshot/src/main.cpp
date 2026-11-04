#define COMPLETION_CPU_POSTSCALE 1
#include <stdio.h>
#include <exception>
#include "hw/pl_intr.h"
#include "completion/completion_graph_runtime.h"
#include "completion/completion_timer.h"


int main(){
	printf("[test] \n");
    setvbuf(stdout,NULL,_IONBF,0);
    printf("[BUILD] daylight-v7 sample=000082_10 original_sgm crop=448,128,576,256\n");
    float probe=1.f;const unsigned char* bytes=(const unsigned char*)&probe;
    printf("[PLATFORM] cpu_hz=%lu float_bytes=%u probe=%02x%02x%02x%02x\n",
       (unsigned long)XPAR_CPU_CORE_CLOCK_FREQ_HZ,(unsigned)sizeof(float),bytes[0],bytes[1],bytes[2],bytes[3]);
    if(sizeof(float)!=4 || bytes[0]!=0 || bytes[1]!=0 || bytes[2]!=128 || bytes[3]!=63) {
        printf("[FAIL] unexpected float format\n");while(1){}
    }
    if(init_intr_sys()!=XST_SUCCESS) {printf("[FAIL] interrupt initialization\n");while(1){}}
    printf("[START] completion\n");
    uint64_t started=0,ended=0;
    const bool start_ok=completion_cycle_snapshot(started);
    int status=8;
    try { status=completion_board_run(); }
    catch(const std::exception& e) { printf("[FAIL] exception: %s\n",e.what()); }
    catch(...) { printf("[FAIL] unknown exception\n"); }
    const bool end_ok=completion_cycle_snapshot(ended);
    if(start_ok && end_ok && ended>started) {
        const uint64_t elapsed=ended-started;
        printf("[PERF] elapsed_cycles_hi=%lu lo=%lu assumed_npu_hz=100000000 includes_node_uart=1\n",
            (unsigned long)(elapsed>>32),(unsigned long)(elapsed&0xffffffffu));
    } else printf("[PERF_UNAVAILABLE] counter unstable, reset, or not advancing; output still available\n");
    printf("[STATUS] %d; output retained for debugger readback\n",status);
    while(1) {}
    return status;
}
