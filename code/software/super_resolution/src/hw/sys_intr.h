
#ifndef __SYS_INTR_H_
#define __SYS_INTR_H_

#include "xparameters.h"
#include "xil_exception.h"
#include "xintc.h"

extern XIntc Intc;

#define INTC_DEVICE_ID XPAR_INTC_0_DEVICE_ID

int Init_Intr_System(XIntc *IntcInstancePtr);
void Setup_Intr_Exception(XIntc *IntcInstancePtr);

#endif
