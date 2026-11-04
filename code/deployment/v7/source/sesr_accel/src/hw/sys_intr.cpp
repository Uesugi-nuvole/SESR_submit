#include "sys_intr.h"

XIntc Intc;

int Init_Intr_System(XIntc *IntcInstancePtr)
{
    int Status;

    Status = XIntc_Initialize(IntcInstancePtr, INTC_DEVICE_ID);
    if (Status != XST_SUCCESS)
    {
        return XST_FAILURE;
    }


    Status = XIntc_Start(IntcInstancePtr, XIN_REAL_MODE);
    if (Status != XST_SUCCESS)
    {
        return XST_FAILURE;
    }

    return XST_SUCCESS;
}

void Setup_Intr_Exception(XIntc *IntcInstancePtr)
{

    Xil_ExceptionInit();


    Xil_ExceptionRegisterHandler(
        XIL_EXCEPTION_ID_INT,
        (Xil_ExceptionHandler)XIntc_InterruptHandler,
        IntcInstancePtr
    );


    Xil_ExceptionEnable();
}
