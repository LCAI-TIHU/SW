#include <aipu_uart.h>
#include <aipu_uart_reg.h>
#include <polling.h>
#include "dla_debug.h"
#include <aipu_io.h>
#include <aipu_parent_clock.h>
#include <metal/compiler.h>
/* This macro enforces that the compiler will not elide the given access. */


static __inline__ uint32_t bauddiv(uint32_t baud, uint64_t clock)
{
    uint32_t div_1000 = 0;
    uint32_t mantissa = 0, fraction = 0;

    div_1000 = (uint64_t)clock * 1000 / (baud * 16);

    mantissa = div_1000 / 1000;
    fraction = div_1000 % 1000 * 16 / 1000;

    if ((div_1000 % 1000 * 16 % 1000) >= 500)
        fraction += 1;

    if (fraction == 16)
    {
        mantissa += 1;
        fraction = 0;
    }

    return ((mantissa << 4) | fraction);
}
static __inline__ void __uart_set_baud_rate(struct aipu_uart *uart, struct metal_clock *sys_clk, int32_t baud_rate)
{
//	debug_trace("enter:%s\n",__func__);
	uint64_t clock_rate = metal_clock_get_rate_hz(sys_clk);
//	debug_trace("clock_rate is %d\n",clock_rate);
	REG_RW(uart->uart_base, CTRL) &= ~(TXEN|RXEN);
	REG_RW(uart->uart_base, BAUDDIV) = bauddiv(baud_rate, clock_rate);
	REG_RW(uart->uart_base, CTRL) |= TXEN;
//	debug_trace("exit:%s\n",__func__);
}
static __inline__ int32_t __uart_check_baud_rate(struct aipu_uart *uart, struct metal_clock *sys_clk, int32_t baud_rate)
{
	uint64_t clock_rate = metal_clock_get_rate_hz(sys_clk);
	if(REG_RW(uart->uart_base, BAUDDIV) == bauddiv(baud_rate, clock_rate))
		return 0;
	else
		return 1;
}
static __inline__ void __uart_init(struct aipu_uart *uart, struct metal_clock *system_clk, int32_t baud_rate)
{
//	debug_trace("enter:%s\n",__func__);
    if (REG_RW(uart->uart_base, REGISTER_BANK_CLOCK_ENABLE_STATUS) == 0)
    {
    	REG_RW(uart->uart_base, REGISTER_BANK_CLOCK_ENABLE_SET) = REGISTER_BANK_CLOCK_ENABLE;
        //while(UART_REG_RW(REGISTER_BANK_CLOCK_ENABLE_STATUS) == 0);
        polling_unexpect_by_count((void *)(UART_REG(uart->uart_base, REGISTER_BANK_CLOCK_ENABLE_STATUS)), 0, 0xFFFFFFFF, 10000);
    }

    if (REG_RW(uart->uart_base, KERNEL_CLOCK_ENABLE_STATUS) == 0)
    {
    	REG_RW(uart->uart_base, KERNEL_CLOCK_ENABLE_SET) = KERNEL_CLOCK_ENABLE;
        //while(UART_REG_RW(KERNEL_CLOCK_ENABLE_STATUS) == 0);
        polling_unexpect_by_count((void *)(UART_REG(uart->uart_base, KERNEL_CLOCK_ENABLE_STATUS)), 0, 0xFFFFFFFF, 10000);
    }

    //TX
    REG_RW(uart->uart_base, TX_FIFO_ENABLE)  = FIFO_EN; //FIFO_EN
    REG_RW(uart->uart_base, TX_FIFO_CONTROL) = FIFO_ABORT; //FIFO ABORT
    REG_RW(uart->uart_base, TX_FIFO_CONTROL) = FIFO_START; //FIFO START
    REG_RW(uart->uart_base, TX_FIFO_CONFIG)  = NON_DMA_CONFIG; //Non-DMA mode

    //RX
    /*
    REG_RW(uart->uart_base, RX_FIFO_ENABLE)  = FIFO_EN; //FIFO_EN
    REG_RW(uart->uart_base, RX_FIFO_CONTROL) = FIFO_ABORT; //FIFO ABORT
    REG_RW(uart->uart_base, RX_FIFO_CONTROL) = FIFO_START; //FIFO START
    REG_RW(uart->uart_base, RX_FIFO_CONFIG)  = NON_DMA_CONFIG; //Non-DMA mode
	*/
    //disable TX/RX when set/change baudrate
    REG_RW(uart->uart_base, CTRL) = CTRL_CONFIG; //1 stop bit, no parity check, 8bit

    /* buadrate */
    __uart_set_baud_rate(uart, system_clk, baud_rate);

    /* TX/RX enable and more like stop bit, parity, data length... */
    REG_RW(uart->uart_base, CTRL) |= TXEN;
    //if use uart, can not print info before here.
}

static __inline__ void __uart_putc(struct aipu_uart *uart, char c)
{
    while(REG_RW(uart->uart_base, TX_FIFO_FULFILL_LEVEL) != 0);
    if(c == '\n')
    	REG_RW(uart->uart_base, TXD_ENTRY_START) = '\r';
    REG_RW(uart->uart_base, TXD_ENTRY_START) = c;

    while(REG_RW(uart->uart_base, TX_FIFO_FULFILL_LEVEL) != 0);
}
static __inline__ int32_t __uart_txready(struct aipu_uart *uart)
{
	if(REG_RW(uart->uart_base, TX_FIFO_FULFILL_LEVEL) != 0)
		return 1;
	else
		return 0;
}

static __inline__ void __uart_getc(struct aipu_uart *uart, char *c)
{
    *(volatile char *)c = (char)REG_RW(uart->uart_base, RXD_ENTRY_START);
}

static __inline__ void __uart_gets(struct aipu_uart *uart, char *addr)
{
	uint32_t i = 0;
    while(REG_RW(uart->uart_base, RX_FIFO_FULFILL_LEVEL) != 0)
    {
             __uart_getc(uart, (char*)(addr+i));
            i++;
    }
}
static __inline__ void __uart_puts(struct aipu_uart *uart, char *s) //for print
{
    while (*s)
    {
    	__uart_putc(uart, *s++);
    }
}
static __inline__ void __uart_transmit_interrupt_enable(struct aipu_uart *uart)
{

}
static __inline__ void __uart_transmit_interrupt_disable(struct aipu_uart *uart)
{

}
static __inline__ void __uart_receive_interrupt_enable(struct aipu_uart *uart)
{

}
static __inline__ void __uart_receive_interrupt_disable(struct aipu_uart *uart)
{

}
static __inline__ void __uart_set_transmit_watermark(struct aipu_uart *uart, size_t level)
{

}
static __inline__ size_t __uart_get_transmit_watermark(struct aipu_uart *uart)
{
	return 0;
}
static __inline__ void __uart_set_receive_watermark(struct aipu_uart *uart, size_t level)
{

}
static __inline__ size_t __uart_get_receive_watermark(struct aipu_uart *uart)
{
	return 0;
}
__METAL_DEFINE_VTABLE(aipu_uart_vtable) = {
    .init = &__uart_init,
    .putc = &__uart_putc,
	.puts = &__uart_puts,
    .getc = &__uart_getc,
	.gets = &__uart_gets,
    .txready = &__uart_txready,
    .check_baud_rate = &__uart_check_baud_rate,
    .set_baud_rate = &__uart_set_baud_rate,
    .tx_interrupt_enable = &__uart_transmit_interrupt_enable,
    .tx_interrupt_disable =
       &__uart_transmit_interrupt_disable,
    .rx_interrupt_enable = &__uart_receive_interrupt_enable,
    .rx_interrupt_disable =
       &__uart_receive_interrupt_disable,
    .set_tx_watermark = &__uart_set_transmit_watermark,
    .get_tx_watermark = &__uart_get_transmit_watermark,
    .set_rx_watermark = &__uart_set_receive_watermark,
    .get_rx_watermark = &__uart_get_receive_watermark,
};

