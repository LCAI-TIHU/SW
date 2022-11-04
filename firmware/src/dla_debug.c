#include <aipu_uart.h>
#include <stdarg.h>
#include <stdio.h>
#include "dla_debug.h"
#include <stdlib.h>

extern struct aipu_uart aipu_uart0;
static char buffer[256];
static char num_buf[16];
static uint64_t info_num = 0;

#ifdef UART_DEBUG
void debug_trace(const char *format, ...){
	int size, i;
	va_list aptr;
	va_start(aptr, format);
	size = vsnprintf(buffer, 255, format, aptr);
	va_end(aptr);

	sprintf(num_buf,"%ld	:", info_num);
	aipu_uart_puts(&aipu_uart0, num_buf);

	for(i = 0; i < size; i++){
		aipu_uart_putc(&aipu_uart0, buffer[i]);
	}
	info_num += 1;
}
#else

void debug_info(const char *format, ...){
	int size, i;
	va_list aptr;
	va_start(aptr, format);
	size = vsnprintf(buffer, 255, format, aptr);
	va_end(aptr);

	//sprintf(num_buf,"%ld	:", info_num);
	//aipu_uart_puts(&aipu_uart0, num_buf);
	//info_num += 1;

	for(i = 0; i < size; i++){
		aipu_uart_putc(&aipu_uart0, buffer[i]);
	}
}

#endif

#ifdef OP_DEBUG
void debug_op(const char *format, ...){
	int size, i;
	va_list aptr;
	va_start(aptr, format);
	size = vsnprintf(buffer, 255, format, aptr);
	va_end(aptr);

//	sprintf(num_buf,"%ld	:", info_num);
//	aipu_uart_puts(&aipu_uart0, num_buf);
//	info_num += 1;

	for(i = 0; i < size; i++){
		aipu_uart_putc(&aipu_uart0, buffer[i]);
	}
}
#endif
void print_float(float_t vec)
{
	if(vec > -1.0 && vec < 0.0)
		debug_info("-");
	debug_info("%d",(int32_t)vec);
	debug_info(".");
	float_t temp = fabs(vec-(int32_t)vec);
	for(int jf = 0; jf < 7; jf++){
		temp = temp*10;
		debug_info("%u",(uint32_t)temp);
		temp = temp - (uint32_t)temp;
	}
	debug_info("\t");
 //    debug_info("\n");
}


#ifdef PCIE_DEBUG
#define PCIE_DEBUG_PTR 0x40800000
static char *buffer = (volatile char *)PCIE_DEBUG_PTR;
void debug_trace(const char *format, ...){
	int  size, i;
	va_list aptr;
	va_start(aptr, format);
	size = vsnprintf(buffer, 255, format, aptr);
//	sifive_pl2cache0_flush((uintptr_t)buffer);
	for(i = 0; i < size; i++){
		aipu_uart_putc(&aipu_uart0, buffer[i]);
	}
	buffer += size;
	va_end(aptr);
}
#endif

#ifdef TIMEOUT_DEBUG
#include "opendla_2048_full.h"//glb test
#include "aipu_io.h"
#include "dla_engine_internal.h"
void time_out_debug(uint64_t *t1, uint64_t *t2){
//	char *test_addr = (char *)malloc(100);
//	debug_info("Test heap is full or not, test_addr = %#x\n", test_addr);
//	free(test_addr);
	debug_info("T1 is %#x, T2 is %#x, T1-T2 is %u\n", *t1, *t2, *t1-*t2);
	debug_info("GLB INTR STATUS IS 0x%08x\n",glb_reg_read(S_INTR_STATUS));
	debug_info("CACC_S_STATUS_0 					0x%08x\n", REG_RW(0x40400000, CACC_S_STATUS_0));
	debug_info("CACC_S_POINTER_0 					0x%08x\n", REG_RW(0x40400000, CACC_S_POINTER_0));
	debug_info("CACC_D_OP_ENABLE_0 				0x%08x\n", REG_RW(0x40400000, CACC_D_OP_ENABLE_0));
	debug_info("CACC_D_MISC_CFG_0 					0x%08x\n", REG_RW(0x40400000, CACC_D_MISC_CFG_0));
	debug_info("CACC_D_DATAOUT_SIZE_0_0 			0x%08x\n", REG_RW(0x40400000, CACC_D_DATAOUT_SIZE_0_0));
	debug_info("CACC_D_DATAOUT_SIZE_1_0 			0x%08x\n", REG_RW(0x40400000, CACC_D_DATAOUT_SIZE_1_0));
	debug_info("CACC_D_DATAOUT_ADDR_0 				0x%08x\n", REG_RW(0x40400000, CACC_D_DATAOUT_ADDR_0));
	debug_info("CACC_D_BATCH_NUMBER_0 				0x%08x\n", REG_RW(0x40400000, CACC_D_BATCH_NUMBER_0));
	debug_info("CACC_D_LINE_STRIDE_0 				0x%08x\n", REG_RW(0x40400000, CACC_D_LINE_STRIDE_0));
	debug_info("CACC_D_SURF_STRIDE_0 				0x%08x\n", REG_RW(0x40400000, CACC_D_SURF_STRIDE_0));
	debug_info("CACC_D_DATAOUT_MAP_0 				0x%08x\n", REG_RW(0x40400000, CACC_D_DATAOUT_MAP_0));
	debug_info("CACC_D_OUT_SATURATION_0 			0x%08x\n", REG_RW(0x40400000, CACC_D_OUT_SATURATION_0));
	REG_RW(0x40400000, SDP_S_POINTER_0) = 0;
	debug_info("SDP_S_STATUS_0 					0x%08x\n", REG_RW(0x40400000, SDP_S_STATUS_0));
	debug_info("SDP_S_POINTER_0 					0x%08x\n", REG_RW(0x40400000, SDP_S_POINTER_0));
	debug_info("SDP_D_OP_ENABLE_0 					0x%08x\n", REG_RW(0x40400000, SDP_D_OP_ENABLE_0));
	debug_info("SDP_D_DATA_CUBE_WIDTH_0 			0x%08x\n", REG_RW(0x40400000, SDP_D_DATA_CUBE_WIDTH_0));
	debug_info("SDP_D_DATA_CUBE_HEIGHT_0 			0x%08x\n", REG_RW(0x40400000, SDP_D_DATA_CUBE_HEIGHT_0));
	debug_info("SDP_D_DATA_CUBE_CHANNEL_0 			0x%08x\n", REG_RW(0x40400000, SDP_D_DATA_CUBE_CHANNEL_0));
	debug_info("SDP_D_DST_BASE_ADDR_LOW_0 			0x%08x\n", REG_RW(0x40400000, SDP_D_DST_BASE_ADDR_LOW_0));
	debug_info("SDP_D_DST_LINE_STRIDE_0 			0x%08x\n", REG_RW(0x40400000, SDP_D_DST_LINE_STRIDE_0));
	debug_info("SDP_D_DST_SURFACE_STRIDE_0 			0x%08x\n", REG_RW(0x40400000, SDP_D_DST_SURFACE_STRIDE_0));
	debug_info("SDP_D_FEATURE_MODE_CFG_0 			0x%08x\n", REG_RW(0x40400000, SDP_D_FEATURE_MODE_CFG_0));
	debug_info("SDP_D_DST_DMA_CFG_0 			0x%08x\n", REG_RW(0x40400000, SDP_D_DST_DMA_CFG_0));
	debug_info("SDP_D_PERF_WDMA_WRITE_STALL_0 			0x%08x\n", REG_RW(0x40400000, SDP_D_PERF_WDMA_WRITE_STALL_0));
	REG_RW(0x40400000, SDP_S_POINTER_1) = 0;
	debug_info("SDP_S_STATUS_1 					0x%08x\n", REG_RW(0x40400000, SDP_S_STATUS_1));
	debug_info("SDP_S_POINTER_1 					0x%08x\n", REG_RW(0x40400000, SDP_S_POINTER_1));
	debug_info("SDP_D_OP_ENABLE_1 					0x%08x\n", REG_RW(0x40400000, SDP_D_OP_ENABLE_1));
	debug_info("SDP_D_DATA_CUBE_WIDTH_1 			0x%08x\n", REG_RW(0x40400000, SDP_D_DATA_CUBE_WIDTH_1));
	debug_info("SDP_D_DATA_CUBE_HEIGHT_1 			0x%08x\n", REG_RW(0x40400000, SDP_D_DATA_CUBE_HEIGHT_1));
	debug_info("SDP_D_DATA_CUBE_CHANNEL_1 			0x%08x\n", REG_RW(0x40400000, SDP_D_DATA_CUBE_CHANNEL_1));
	debug_info("SDP_D_DST_BASE_ADDR_LOW_1 			0x%08x\n", REG_RW(0x40400000, SDP_D_DST_BASE_ADDR_LOW_1));
	debug_info("SDP_D_DST_LINE_STRIDE_1 			0x%08x\n", REG_RW(0x40400000, SDP_D_DST_LINE_STRIDE_1));
	debug_info("SDP_D_DST_SURFACE_STRIDE_1 			0x%08x\n", REG_RW(0x40400000, SDP_D_DST_SURFACE_STRIDE_1));
	debug_info("SDP_D_FEATURE_MODE_CFG_1 			0x%08x\n", REG_RW(0x40400000, SDP_D_FEATURE_MODE_CFG_1));
	debug_info("SDP_D_DST_DMA_CFG_1 			0x%08x\n", REG_RW(0x40400000, SDP_D_DST_DMA_CFG_1));
	debug_info("SDP_D_PERF_WDMA_WRITE_STALL_1 			0x%08x\n", REG_RW(0x40400000, SDP_D_PERF_WDMA_WRITE_STALL_1));
	debug_info("CSC_S_STATUS_1 			0x%08x\n", REG_RW(0x40400000, CSC_S_STATUS_1));
	debug_info("CSC_S_POINTER_1 			0x%08x\n", REG_RW(0x40400000, CSC_S_POINTER_1));
	debug_info("CSC_D_OP_ENABLE_1 			0x%08x\n", REG_RW(0x40400000, CSC_D_OP_ENABLE_1));
	debug_info("CDMA_S_STATUS_1 			0x%08x\n", REG_RW(0x40400000, CDMA_S_STATUS_1));
	debug_info("CDMA_S_POINTER_1 			0x%08x\n", REG_RW(0x40400000, CDMA_S_POINTER_1));
	debug_info("CDMA_D_OP_ENABLE_1 			0x%08x\n", REG_RW(0x40400000, CDMA_D_OP_ENABLE_1));
	debug_info("CDMA_D_DATAIN_SIZE_0_1 			0x%08x\n", REG_RW(0x40400000, CDMA_D_DATAIN_SIZE_0_1));
	debug_info("CDMA_D_WEIGHT_SIZE_0_1 			0x%08x\n", REG_RW(0x40400000, CDMA_D_WEIGHT_SIZE_0_1));
	/*----------------------TEST PERFORMANCE----------------------*/
//	debug_info("GLB_S_CDMA_DAT_COUNTER 			0x%08x\n", REG_RW(0x40400000, GLB_S_CDMA_DAT_COUNTER));
//	debug_info("GLB_S_CDMA_WT_COUNTER 			0x%08x\n", REG_RW(0x40400000, GLB_S_CDMA_WT_COUNTER));
//	debug_info("GLB_S_CDMA_OP_EN_START_COUNTER 			0x%08x\n", REG_RW(0x40400000, GLB_S_CDMA_OP_EN_START_COUNTER));
//	debug_info("GLB_S_CDMA_OP_EN_COUNTER 			0x%08x\n", REG_RW(0x40400000, GLB_S_CDMA_OP_EN_COUNTER));
//	debug_info("GLB_S_CACC_PER_COUNTER 			0x%08x\n", REG_RW(0x40400000, GLB_S_CACC_PER_COUNTER));
//	debug_info("GLB_S_SDP_PER_COUNTER 			0x%08x\n", REG_RW(0x40400000, GLB_S_SDP_PER_COUNTER));
//	debug_info("GLB_S_PDP_PER_COUNTER 			0x%08x\n", REG_RW(0x40400000, GLB_S_PDP_PER_COUNTER));


}
#endif
