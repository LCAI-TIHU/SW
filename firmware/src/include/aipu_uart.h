/*
 * aipu_uart.h
 *
 *  Created on: Nov 30, 2021
 *      Author: root
 */

#ifndef AIPU_UART_H_
#define AIPU_UART_H_
#include <metal/interrupt.h>
#include <metal/clock.h>
#include <metal/compiler.h>
#define UART0_BASE 									   0x40100000
#define UART0_BAUD_RATE								   115200UL

#define UART_REG(base, offset) ((unsigned long)base + offset)
#define REGISTER_BANK_CLOCK_ENABLE 		0x00000001
#define KERNEL_CLOCK_ENABLE 			0x00000001
#define FIFO_EN 						0x00000004
#define FIFO_ABORT						0x00000001
#define FIFO_START						0x00000002
#define NON_DMA_CONFIG					0x00000000
#define CTRL_CONFIG						0x00000C00
#define TXEN							0x00000001
#define RXEN							0x00000002


struct aipu_uart;
#undef getc
#undef putc
#undef puts
#undef gets
struct aipu_uart_vtable {
    void (*init)(struct aipu_uart *uart, struct metal_clock *system_clk, int baud_rate);
    void (*putc)(struct aipu_uart *uart, char c);
    void (*puts)(struct aipu_uart *uart, char *s);
    int (*txready)(struct aipu_uart *uart);
    void (*getc)(struct aipu_uart *uart, char *c);
    void (*gets)(struct aipu_uart *uart, char *s);
    int (*check_baud_rate)(struct aipu_uart *uart, struct metal_clock *sys_clk, int baud_rate);
    void (*set_baud_rate)(struct aipu_uart *uart,  struct metal_clock *sys_clk, int baud_rate);
    void (*tx_interrupt_enable)(struct aipu_uart *uart);
    void (*tx_interrupt_disable)(struct aipu_uart *uart);
    void (*rx_interrupt_enable)(struct aipu_uart *uart);
    void (*rx_interrupt_disable)(struct aipu_uart *uart);
    void (*set_tx_watermark)(struct aipu_uart *uart, size_t length);
    size_t (*get_tx_watermark)(struct aipu_uart *uart);
    void (*set_rx_watermark)(struct aipu_uart *uart, size_t length);
    size_t (*get_rx_watermark)(struct aipu_uart *uart);
};

/*!
 * @brief Handle for a UART serial device
 */
struct aipu_uart {
    const struct aipu_uart_vtable *vtable;
    long uart_base;
//    int tx_interrupt_id;
//    int rx_interrupt_id;
};
__METAL_DECLARE_VTABLE(aipu_uart_vtable);

/*!
 * @brief Initialize UART device

 * Initialize the UART device described by the UART handle. This function must
 be called before any
 * other method on the UART can be invoked. It is invalid to initialize a UART
 more than once.
 *
 * @param uart The UART device handle
 * @param baud_rate the baud rate to set the UART to
 */
static __inline__ void aipu_uart_init(struct aipu_uart *uart, struct metal_clock *system_clk, int baud_rate) {
    uart->vtable->init(uart, system_clk, baud_rate);
}

/*!
 * @brief Output a character over the UART
 * @param uart The UART device handle
 * @param c The character to send over the UART
 * @return 0 upon success
 */
static __inline__ void aipu_uart_putc(struct aipu_uart *uart, char c) {
    uart->vtable->putc(uart, c);
}
static __inline__ void aipu_uart_puts(struct aipu_uart *uart, char *s) {
    uart->vtable->puts(uart, s);
}

/*!
 * @brief Test, determine if tx output is blocked(full/busy)
 * @param uart The UART device handle
 * @return 0 not blocked
 */
static __inline__ int aipu_uart_txready(struct aipu_uart *uart) {
    return uart->vtable->txready(uart);
}

/*!
 * @brief Read a character sent over the UART
 * @param uart The UART device handle
 * @param c The varible to hold the read character
 * @return 0 upon success
 *
 * If "c == -1" no char was ready.
 * If "c != -1" then C == byte value (0x00 to 0xff)
 */
static __inline__ void aipu_uart_getc(struct aipu_uart *uart, char *c) {
    uart->vtable->getc(uart, c);
}
static __inline__ void aipu_uart_gets(struct aipu_uart *uart, char *s) {
    uart->vtable->gets(uart, s);
}

/*!
 * @brief Get the baud rate of the UART peripheral
 * @param uart The UART device handle
 * @return The current baud rate of the UART
 */
static __inline__ int aipu_uart_check_baud_rate(struct aipu_uart *uart,
		struct metal_clock *sys_clk, int baud_rate) {
    return uart->vtable->check_baud_rate(uart, sys_clk, baud_rate);
}

/*!
 * @brief Set the baud rate of the UART peripheral
 * @param uart The UART device handle
 * @param baud_rate The baud rate to configure
 * @return the new baud rate of the UART
 */
static __inline__ void aipu_uart_set_baud_rate(struct aipu_uart *uart,
		 struct metal_clock *sys_clk, int baud_rate) {
    uart->vtable->set_baud_rate(uart, sys_clk, baud_rate);
}

/*!
 * @brief Enable the UART transmit interrupt
 * @param uart The UART device handle
 * @return 0 upon success
 */
static __inline__ void aipu_uart_transmit_interrupt_enable(struct aipu_uart *uart) {
    uart->vtable->tx_interrupt_enable(uart);
}

/*!
 * @brief Disable the UART transmit interrupt
 * @param uart The UART device handle
 * @return 0 upon success
 */
static __inline__ void aipu_uart_transmit_interrupt_disable(struct aipu_uart *uart) {
    uart->vtable->tx_interrupt_disable(uart);
}

/*!
 * @brief Enable the UART receive interrupt
 * @param uart The UART device handle
 * @return 0 upon success
 */
static __inline__ void aipu_uart_receive_interrupt_enable(struct aipu_uart *uart) {
    uart->vtable->rx_interrupt_enable(uart);
}

/*!
 * @brief Disable the UART receive interrupt
 * @param uart The UART device handle
 * @return 0 upon success
 */
__inline__ void aipu_uart_receive_interrupt_disable(struct aipu_uart *uart) {
    uart->vtable->rx_interrupt_disable(uart);
}

/*!
 * @brief Set the transmit watermark level of the UART controller
 * @param uart The UART device handle
 * @param level The UART transmit watermark level
 * @return 0 upon success
 */
static __inline__ void aipu_uart_set_transmit_watermark(struct aipu_uart *uart,
                                                 size_t level) {
    uart->vtable->set_tx_watermark(uart, level);
}

/*!
 * @brief Get the transmit watermark level of the UART controller
 * @param uart The UART device handle
 * @return The UART transmit watermark level
 */
static __inline__ size_t aipu_uart_get_transmit_watermark(struct aipu_uart *uart) {
    return uart->vtable->get_tx_watermark(uart);
}

/*!
 * @brief Set the receive watermark level of the UART controller
 * @param uart The UART device handle
 * @param level The UART transmit watermark level
 * @return 0 upon success
 */
static __inline__ void aipu_uart_set_receive_watermark(struct aipu_uart *uart,
                                                size_t level) {
    uart->vtable->set_rx_watermark(uart, level);
}

/*!
 * @brief Get the receive watermark level of the UART controller
 * @param uart The UART device handle
 * @return The UART transmit watermark level
 */
static __inline__ size_t aipu_uart_get_receive_watermark(struct aipu_uart *uart) {
    return uart->vtable->get_rx_watermark(uart);
}



#endif /* aipu_UART_H_ */
