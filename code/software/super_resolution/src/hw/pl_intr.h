#ifndef __PL_INTR_H_
#define __PL_INTR_H_

#include <stdio.h>
#include "xparameters.h"
#include "xintc.h"
#include "xil_exception.h"
#include "xstatus.h"
#include "sys_intr.h"

/*
 * MicroBlaze + AXI INTC ÏÂ£¬²»Ê¹ÓÃ XScuGic¡£
 * ÖÐ¶ÏºÅÊ¹ÓÃ xparameters.h ×Ô¶¯Éú³ÉµÄ AXI INTC ÊäÈë±àºÅ¡£
 */
#define ALU_INT_ID   XPAR_AXI_INTC_0_TCP_0_IO_ALU_TASK_DONE_INTR
#define GEMM_INT_ID  XPAR_AXI_INTC_0_TCP_0_IO_GEMM_TASK_DONE_INTR
#define POOL_INT_ID  XPAR_AXI_INTC_0_TCP_0_IO_POOL_TASK_DONE_INTR
#define PIXELSHUFFLE_INT_ID XPAR_AXI_INTC_0_TCP_0_IO_PIXELSHUFFLE_TASK_DONE_INTR
#define IMAGE_INT_ID  XPAR_AXI_INTC_0_TCP_0_IO_SESR_IMAGE_TASK_DONE_INTR

/*
 * Èç¹ûÄãÏë±£ÁôÔ­À´µÄ SW1/SW2/SW3 ÃüÃû£¬Ò²¿ÉÒÔÕâÑùÓ³Éä¡£
 * ×¢Òâ£ºÕâÀïµÄ SW1/SW2/SW3 ÒÑ¾­²»ÊÇ ARM GIC µÄ 121/122/123£¬
 * ¶øÊÇ AXI INTC µÄÊäÈë±àºÅ 0/1/2¡£
 */
#define SW1_INT_ID   ALU_INT_ID
#define SW2_INT_ID   POOL_INT_ID
#define SW3_INT_ID   GEMM_INT_ID
#define SW4_INT_ID   PIXELSHUFFLE_INT_ID
#define SW5_INT_ID   IMAGE_INT_ID
/*
 * Äãµ±Ç° xparameters.h ÀïÃæÃ»ÓÐ SW4/SW5 ¶ÔÓ¦µÄÖÐ¶Ïºê£¬
 * ËùÒÔÏÈ²»Òª¶¨Òå SW4_INT_ID/SW5_INT_ID¡£
 */

/*
 * ÖÐ¶Ï±êÖ¾½¨Òé¼Ó volatile¡£
 * ÒòÎªÕâÐ©±äÁ¿»áÔÚÖÐ¶Ïº¯ÊýÀï±»ÐÞ¸Ä£¬ÔÚÖ÷Ñ­»·Àï±»¶ÁÈ¡¡£
 */
extern volatile int alu_mat_task_done;
extern volatile int pool_task_done;
extern volatile int gemm_task_done;
extern volatile int pixelshuffle_task_done;
extern volatile int image_task_done;


void SW1_intr_Handler(void *param);
void SW2_intr_Handler(void *param);
void SW3_intr_Handler(void *param);
void SW4_intr_Handler(void *param);
void SW5_intr_Handler(void *param);

int pl_intr_init(XIntc *InstancePtr);
void init_intr_sys(void);

#endif
