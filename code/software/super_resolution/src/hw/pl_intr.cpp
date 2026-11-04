#include "pl_intr.h"
#include "xil_io.h"

/*
 * ÖÐ¶Ï±êÖ¾±äÁ¿¡£
 * ¼Ó volatile£¬±ÜÃâ±àÒëÆ÷ÓÅ»¯µ¼ÖÂÖ÷Ñ­»·¶Á²»µ½ÖÐ¶ÏÀïÐÞ¸ÄºóµÄÖµ¡£
 */
volatile int alu_mat_task_done = 0;
volatile int pool_task_done = 0;
volatile int gemm_task_done = 0;
//volatile int hdmi_input_flag = 0;
volatile int pixelshuffle_task_done = 0;
volatile int image_task_done = 0;

/*
 * ³õÊ¼»¯Õû¸öÖÐ¶ÏÏµÍ³
 */
void init_intr_sys(void)
{
    int status;


    status = Init_Intr_System(&Intc);
    if (status != XST_SUCCESS) {
        printf("[ error ]: Init_Intr_System failed\r\n");
        return;
    }


    status = pl_intr_init(&Intc);
    if (status != XST_SUCCESS) {
        printf("[ error ]: pl_intr_init failed\r\n");
        return;
    }


    status = XIntc_Start(&Intc, XIN_REAL_MODE);
    if (status != XST_SUCCESS) {
        printf("[ error ]: XIntc_Start failed\r\n");
        return;
    }


    Setup_Intr_Exception(&Intc);

}


/*
 * ALU task done ÖÐ¶Ï
 */
void SW1_intr_Handler(void *param)
{
    alu_mat_task_done = 1;


}


/*
 * POOL task done ÖÐ¶Ï
 */
void SW2_intr_Handler(void *param)
{
    pool_task_done = 1;

}


/*
 * GEMM task done ÖÐ¶Ï
 */
void SW3_intr_Handler(void *param)
{
    gemm_task_done = 1;
}

//pixelshuffle task done
void SW4_intr_Handler(void *param)
{
	pixelshuffle_task_done = 1;
}

void SW5_intr_Handler(void *param)
{
	image_task_done = 1;
}

int pl_intr_init(XIntc *InstancePtr)
{
    int status;

    status = XIntc_Connect(InstancePtr,
                           SW1_INT_ID,
                           (XInterruptHandler)SW1_intr_Handler,
                           (void *)1);
    if (status != XST_SUCCESS) {
        printf("[ error ]: XIntc_Connect SW1/ALU failed\r\n");
        return XST_FAILURE;
    }


    status = XIntc_Connect(InstancePtr,
                           SW2_INT_ID,
                           (XInterruptHandler)SW2_intr_Handler,
                           (void *)2);
    if (status != XST_SUCCESS) {
        printf("[ error ]: XIntc_Connect SW2/POOL failed\r\n");
        return XST_FAILURE;
    }


    status = XIntc_Connect(InstancePtr,
                           SW3_INT_ID,
                           (XInterruptHandler)SW3_intr_Handler,
                           (void *)3);
    if (status != XST_SUCCESS) {
        printf("[ error ]: XIntc_Connect SW3/GEMM failed\r\n");
        return XST_FAILURE;
    }

    status = XIntc_Connect(InstancePtr,
    		                SW4_INT_ID,
                            (XInterruptHandler)SW4_intr_Handler,
                            (void *)4);
    if (status != XST_SUCCESS) {
            printf("[ error ]: XIntc_Connect SW4/PIXELSHUFFLE failed\r\n");
            return XST_FAILURE;
    }

    status = XIntc_Connect(InstancePtr,
        		                SW5_INT_ID,
                                (XInterruptHandler)SW5_intr_Handler,
                                (void *)5);
        if (status != XST_SUCCESS) {
                printf("[ error ]: XIntc_Connect SW5/IMAGE failed\r\n");
                return XST_FAILURE;
        }


    XIntc_Enable(InstancePtr, SW1_INT_ID);
    XIntc_Enable(InstancePtr, SW2_INT_ID);
    XIntc_Enable(InstancePtr, SW3_INT_ID);
    XIntc_Enable(InstancePtr, SW4_INT_ID);
    XIntc_Enable(InstancePtr, SW5_INT_ID);

    return XST_SUCCESS;
}
