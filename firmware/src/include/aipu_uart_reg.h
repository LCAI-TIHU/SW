#ifndef AIPU_UART_REG_H
#define AIPU_UART_REG_H

#define REGISTER_BANK_CLOCK_ENABLE_SET                 0x00000000
#define REGISTER_BANK_CLOCK_ENABLE_CLR                 0x00000004
#define REGISTER_BANK_CLOCK_ENABLE_STATUS              0x00000008
#define KERNEL_CLOCK_ENABLE_SET                        0x0000000C
#define KERNEL_CLOCK_ENABLE_CLR                        0x00000010
#define KERNEL_CLOCK_ENABLE_STATUS                     0x00000014

#define CHIPID                                         0x00000018
#define MODULEID                                       0x0000001C
#define CHECKSUM                                       0x00000020

#define FUNC_INT_0_ENABLE_SET                          0x00000024
#define FUNC_INT_0_ENABLE_CLEAR                        0x00000028
#define FUNC_INT_0_ENABLE_STATUS                       0x0000002C
#define FUNC_INT_0_SET                                 0x00000030
#define FUNC_INT_0_CLR                                 0x00000034
#define FUNC_INT_0_STATUS_ENABLED                      0x00000038
#define FUNC_INT_0_STATUS                              0x0000003C

#define TX_FIFO_ENABLE                                 0x00000078
#define TX_FIFO_COMMON_INT_ENABLE                      0x0000007C
#define TX_FIFO_COMMON_INT_SET                         0x00000080
#define TX_FIFO_COMMON_INT_CLR                         0x00000084
#define TX_FIFO_COMMON_INT_STATUS_ENABLED              0x00000088
#define TX_FIFO_COMMON_INT_STATUS                      0x0000008C
#define TX_FIFO_CONTROL                                0x00000090
#define TX_FIFO_CONFIG                                 0x00000094
#define TX_FIFO_FULFILL_LEVEL                          0x00000098
#define TX_FIFO_FSM_STATUS                             0x0000009C

#define RX_FIFO_ENABLE                                 0x000000A0
#define RX_FIFO_COMMON_INT_ENABLE                      0x000000A4
#define RX_FIFO_COMMON_INT_SET                         0x000000A8
#define RX_FIFO_COMMON_INT_CLR                         0x000000AC
#define RX_FIFO_COMMON_INT_STATUS_ENABLED              0x000000B0
#define RX_FIFO_COMMON_INT_STATUS                      0x000000B4
#define RX_FIFO_CONTROL                                0x000000B8
#define RX_FIFO_CONFIG                                 0x000000BC
#define RX_FIFO_FULFILL_LEVEL                          0x000000C0
#define RX_FIFO_FSM_STATUS                             0x000000C4

#define TXD_ENTRY_START                                0x00000400
#define TXD_ENTRY_END                                  0x000007FC

#define RXD_ENTRY_START                                0x00000800
#define RXD_ENTRY_END                                  0x00000BFC

#define CTRL                                           0x00000C00
#define BAUDDIV                                        0x00000C04

#define RXSTATE					       0x00000C08
#define TXSTATE					       0x00000C0C

#define TXD_ENTRY_FULL									(TXD_ENTRY_END-TXD_ENTRY_START)
#define RXD_ENTRY_FULL									(RXD_ENTRY_END-RXD_ENTRY_START)
#endif
