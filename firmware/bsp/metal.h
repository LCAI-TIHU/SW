/* Copyright 2019 SiFive, Inc */
/* SPDX-License-Identifier: Apache-2.0 */
/* ----------------------------------- */
/* ----------------------------------- */

#ifndef ASSEMBLY

#include <metal/machine/platform.h>

#ifdef __METAL_MACHINE_MACROS

#ifndef MACROS_IF_METAL_H
#define MACROS_IF_METAL_H

#define __METAL_CLINT_NUM_PARENTS 2

#ifndef __METAL_CLINT_NUM_PARENTS
#define __METAL_CLINT_NUM_PARENTS 0
#endif
#define __METAL_PLIC_SUBINTERRUPTS 64

#define __METAL_PLIC_NUM_PARENTS 2

#ifndef __METAL_PLIC_SUBINTERRUPTS
#define __METAL_PLIC_SUBINTERRUPTS 0
#endif
#ifndef __METAL_PLIC_NUM_PARENTS
#define __METAL_PLIC_NUM_PARENTS 0
#endif
#ifndef __METAL_CLIC_SUBINTERRUPTS
#define __METAL_CLIC_SUBINTERRUPTS 0
#endif

#endif /* MACROS_IF_METAL_H*/

#else /* ! __METAL_MACHINE_MACROS */

#ifndef MACROS_ELSE_METAL_H
#define MACROS_ELSE_METAL_H

#define __METAL_CLINT_2000000_INTERRUPTS 2

#define METAL_MAX_CLINT_INTERRUPTS 2

#define __METAL_CLINT_NUM_PARENTS 2

#define __METAL_INTERRUPT_CONTROLLER_C000000_INTERRUPTS 2

#define __METAL_PLIC_SUBINTERRUPTS 64

#define METAL_MAX_PLIC_INTERRUPTS 2

#define __METAL_PLIC_NUM_PARENTS 2

#define __METAL_CLIC_SUBINTERRUPTS 0
#define METAL_MAX_CLIC_INTERRUPTS 0

#define METAL_MAX_LOCAL_EXT_INTERRUPTS 0

#define __METAL_GLOBAL_EXTERNAL_INTERRUPTS_INTERRUPTS 63

#define METAL_MAX_GLOBAL_EXT_INTERRUPTS 63

#define METAL_MAX_GPIO_INTERRUPTS 0

#define METAL_MAX_I2C0_INTERRUPTS 0

#define METAL_SIFIVE_L2PF1_BASE_ADDR {\
	METAL_SIFIVE_L2PF1_0_BASE_ADDRESS,\
	}

#define METAL_SIFIVE_L2PF1_QUEUE_ENTRIES {\
	METAL_SIFIVE_L2PF1_0_QUEUE_ENTRIES,\
	}

#define METAL_SIFIVE_L2PF1_WINDOW_BITS {\
	METAL_SIFIVE_L2PF1_0_WINDOW_BITS,\
	}

#define METAL_SIFIVE_L2PF1_DISTANCE_BITS {\
	METAL_SIFIVE_L2PF1_0_DISTANCE_BITS,\
	}

#define METAL_SIFIVE_L2PF1_STREAMS {\
	METAL_SIFIVE_L2PF1_0_STREAMS,\
	}

#define METAL_SIFIVE_L2PF1_0_QUEUE_ENTRIES 16

#define METAL_SIFIVE_L2PF1_0_WINDOW_BITS 6

#define METAL_SIFIVE_L2PF1_0_DISTANCE_BITS 6

#define METAL_SIFIVE_L2PF1_0_STREAMS 8

#define METAL_PL2CACHE_DRIVER_PREFIX sifive_pl2cache0

#define METAL_SIFIVE_PL2CACHE0_BASE_ADDR {\
				METAL_SIFIVE_PL2CACHE0_0_BASE_ADDRESS,\
				}

#define METAL_SIFIVE_PL2CACHE0_PERFMON_COUNTERS 6

#define __METAL_DT_SIFIVE_PL2CACHE0_HANDLE (struct metal_cache *)NULL

#define METAL_MAX_PWM0_INTERRUPTS 0

#define METAL_MAX_PWM0_NCMP 0

#define METAL_MAX_UART_INTERRUPTS 0

#define METAL_MAX_SIMUART_INTERRUPTS 0


#include <metal/drivers/fixed-clock.h>
#include <metal/memory.h>
#include <metal/drivers/riscv_clint0.h>
#include <metal/drivers/riscv_cpu.h>
#include <metal/drivers/riscv_plic0.h>
#include <metal/pmp.h>
#include <metal/drivers/sifive_global-external-interrupts0.h>
/* #include <metal/drivers/sifive_l2pf1.h> */
/* #include <metal/drivers/sifive_pl2cache0.h> */
#include <metal/drivers/sifive_test0.h>

/* From subsystem_pbus_clock */

extern struct __metal_driver_fixed_clock __metal_dt_subsystem_pbus_clock;

extern struct metal_memory __metal_dt_mem_testram_40000000;

extern struct metal_memory __metal_dt_mem_memory_80000000;

/* From clint@2000000 */
extern struct __metal_driver_riscv_clint0 __metal_dt_clint_2000000;

/* From cpu@0 */
extern struct __metal_driver_cpu __metal_dt_cpu_0;

extern struct __metal_driver_riscv_cpu_intc __metal_dt_cpu_0_interrupt_controller;

/* From interrupt_controller@c000000 */
extern struct __metal_driver_riscv_plic0 __metal_dt_interrupt_controller_c000000;

extern struct metal_pmp __metal_dt_pmp;

/* From global_external_interrupts */
extern struct __metal_driver_sifive_global_external_interrupts0 __metal_dt_global_external_interrupts;

/* From teststatus@4000 */
extern struct __metal_driver_sifive_test0 __metal_dt_teststatus_4000;



/* --------------------- fixed_clock ------------ */
static __inline__ unsigned long __metal_driver_fixed_clock_rate(const struct metal_clock *clock)
{
	if ((uintptr_t)clock == (uintptr_t)&__metal_dt_subsystem_pbus_clock) {
		return METAL_FIXED_CLOCK__SUBSYSTEM_PBUS_CLOCK_CLOCK_FREQUENCY;
	}
	else {
		return 0;
	}
}



/* --------------------- fixed_factor_clock ------------ */


/* --------------------- sifive_clint0 ------------ */
static __inline__ unsigned long __metal_driver_sifive_clint0_control_base(struct metal_interrupt *controller)
{
	if ((uintptr_t)controller == (uintptr_t)&__metal_dt_clint_2000000) {
		return METAL_RISCV_CLINT0_2000000_BASE_ADDRESS;
	}
	else {
		return 0;
	}
}

static __inline__ unsigned long __metal_driver_sifive_clint0_control_size(struct metal_interrupt *controller)
{
	if ((uintptr_t)controller == (uintptr_t)&__metal_dt_clint_2000000) {
		return METAL_RISCV_CLINT0_2000000_SIZE;
	}
	else {
		return 0;
	}
}

static __inline__ int __metal_driver_sifive_clint0_num_interrupts(struct metal_interrupt *controller)
{
	if ((uintptr_t)controller == (uintptr_t)&__metal_dt_clint_2000000) {
		return METAL_MAX_CLINT_INTERRUPTS;
	}
	else {
		return 0;
	}
}

static __inline__ struct metal_interrupt * __metal_driver_sifive_clint0_interrupt_parents(struct metal_interrupt *controller, int idx)
{
	if (idx == 0) {
		return (struct metal_interrupt *)&__metal_dt_cpu_0_interrupt_controller.controller;
	}
	else if (idx == 1) {
		return (struct metal_interrupt *)&__metal_dt_cpu_0_interrupt_controller.controller;
	}
	else {
		return NULL;
	}
}

static __inline__ int __metal_driver_sifive_clint0_interrupt_lines(struct metal_interrupt *controller, int idx)
{
	if (idx == 0) {
		return 3;
	}
	else if (idx == 1) {
		return 7;
	}
	else {
		return 0;
	}
}



/* --------------------- cpu ------------ */
static __inline__ int __metal_driver_cpu_hartid(struct metal_cpu *cpu)
{
	if ((uintptr_t)cpu == (uintptr_t)&__metal_dt_cpu_0) {
		return 0;
	}
	else {
		return -1;
	}
}

static __inline__ int __metal_driver_cpu_timebase(struct metal_cpu *cpu)
{
	if ((uintptr_t)cpu == (uintptr_t)&__metal_dt_cpu_0) {
		return 1000000;
	}
	else {
		return 0;
	}
}

static __inline__ struct metal_interrupt * __metal_driver_cpu_interrupt_controller(struct metal_cpu *cpu)
{
	if ((uintptr_t)cpu == (uintptr_t)&__metal_dt_cpu_0) {
		return &__metal_dt_cpu_0_interrupt_controller.controller;
	}
	else {
		return NULL;
	}
}

static __inline__ int __metal_driver_cpu_num_pmp_regions(struct metal_cpu *cpu)
{
	if ((uintptr_t)cpu == (uintptr_t)&__metal_dt_cpu_0) {
		return 1;
	}
	else {
		return 0;
	}
}

static __inline__ struct metal_buserror * __metal_driver_cpu_buserror(struct metal_cpu *cpu)
{
	if ((uintptr_t)cpu == (uintptr_t)&__metal_dt_cpu_0) {
		return NULL;
	}
	else {
		return NULL;
	}
}



/* --------------------- sifive_plic0 ------------ */
static __inline__ unsigned long __metal_driver_sifive_plic0_control_base(struct metal_interrupt *controller)
{
	if ((uintptr_t)controller == (uintptr_t)&__metal_dt_interrupt_controller_c000000) {
		return METAL_RISCV_PLIC0_C000000_BASE_ADDRESS;
	}
	else {
		return 0;
	}
}

static __inline__ unsigned long __metal_driver_sifive_plic0_control_size(struct metal_interrupt *controller)
{
	if ((uintptr_t)controller == (uintptr_t)&__metal_dt_interrupt_controller_c000000) {
		return METAL_RISCV_PLIC0_C000000_SIZE;
	}
	else {
		return 0;
	}
}

static __inline__ int __metal_driver_sifive_plic0_num_interrupts(struct metal_interrupt *controller)
{
	if ((uintptr_t)controller == (uintptr_t)&__metal_dt_interrupt_controller_c000000) {
		return METAL_RISCV_PLIC0_C000000_RISCV_NDEV;
	}
	else {
		return 0;
	}
}

static __inline__ int __metal_driver_sifive_plic0_max_priority(struct metal_interrupt *controller)
{
	if ((uintptr_t)controller == (uintptr_t)&__metal_dt_interrupt_controller_c000000) {
		return METAL_RISCV_PLIC0_C000000_RISCV_MAX_PRIORITY;
	}
	else {
		return 0;
	}
}

static __inline__ struct metal_interrupt * __metal_driver_sifive_plic0_interrupt_parents(struct metal_interrupt *controller, int idx)
{
	if (idx == 0) {
		return (struct metal_interrupt *)&__metal_dt_cpu_0_interrupt_controller.controller;
	}
	else if (idx == 1) {
		return (struct metal_interrupt *)&__metal_dt_cpu_0_interrupt_controller.controller;
	}
	else {
		return NULL;
	}
}

static __inline__ int __metal_driver_sifive_plic0_interrupt_lines(struct metal_interrupt *controller, int idx)
{
	if (idx == 0) {
		return 11;
	}
	else if (idx == 1) {
		return 9;
	}
	else {
		return 0;
	}
}

static __inline__ int __metal_driver_sifive_plic0_context_ids(int hartid)
{
	if (hartid == 0) {
		return 0;
	}
	else {
		return -1;
	}
}



/* --------------------- sifive_buserror0 ------------ */


/* --------------------- sifive_clic0 ------------ */


/* --------------------- sifive_local_external_interrupts0 ------------ */


/* --------------------- sifive_global_external_interrupts0 ------------ */
static __inline__ int __metal_driver_sifive_global_external_interrupts0_init_done()
{
		return 0;
}

static __inline__ struct metal_interrupt * __metal_driver_sifive_global_external_interrupts0_interrupt_parent(struct metal_interrupt *controller)
{
	if ((uintptr_t)controller == (uintptr_t)&__metal_dt_global_external_interrupts) {
		return (struct metal_interrupt *)&__metal_dt_interrupt_controller_c000000.controller;
	}
	else {
		return NULL;
	}
}

static __inline__ int __metal_driver_sifive_global_external_interrupts0_num_interrupts(struct metal_interrupt *controller)
{
	if ((uintptr_t)controller == (uintptr_t)&__metal_dt_global_external_interrupts) {
		return METAL_MAX_GLOBAL_EXT_INTERRUPTS;
	}
	else {
		return 0;
	}
}

static __inline__ int __metal_driver_sifive_global_external_interrupts0_interrupt_lines(struct metal_interrupt *controller, int idx)
{
	if (idx == 0) {
		return 1;
	}
	else if (idx == 1) {
		return 2;
	}
	else if (idx == 2) {
		return 3;
	}
	else if (idx == 3) {
		return 4;
	}
	else if (idx == 4) {
		return 5;
	}
	else if (idx == 5) {
		return 6;
	}
	else if (idx == 6) {
		return 7;
	}
	else if (idx == 7) {
		return 8;
	}
	else if (idx == 8) {
		return 9;
	}
	else if (idx == 9) {
		return 10;
	}
	else if (idx == 10) {
		return 11;
	}
	else if (idx == 11) {
		return 12;
	}
	else if (idx == 12) {
		return 13;
	}
	else if (idx == 13) {
		return 14;
	}
	else if (idx == 14) {
		return 15;
	}
	else if (idx == 15) {
		return 16;
	}
	else if (idx == 16) {
		return 17;
	}
	else if (idx == 17) {
		return 18;
	}
	else if (idx == 18) {
		return 19;
	}
	else if (idx == 19) {
		return 20;
	}
	else if (idx == 20) {
		return 21;
	}
	else if (idx == 21) {
		return 22;
	}
	else if (idx == 22) {
		return 23;
	}
	else if (idx == 23) {
		return 24;
	}
	else if (idx == 24) {
		return 25;
	}
	else if (idx == 25) {
		return 26;
	}
	else if (idx == 26) {
		return 27;
	}
	else if (idx == 27) {
		return 28;
	}
	else if (idx == 28) {
		return 29;
	}
	else if (idx == 29) {
		return 30;
	}
	else if (idx == 30) {
		return 31;
	}
	else if (idx == 31) {
		return 32;
	}
	else if (idx == 32) {
		return 33;
	}
	else if (idx == 33) {
		return 34;
	}
	else if (idx == 34) {
		return 35;
	}
	else if (idx == 35) {
		return 36;
	}
	else if (idx == 36) {
		return 37;
	}
	else if (idx == 37) {
		return 38;
	}
	else if (idx == 38) {
		return 39;
	}
	else if (idx == 39) {
		return 40;
	}
	else if (idx == 40) {
		return 41;
	}
	else if (idx == 41) {
		return 42;
	}
	else if (idx == 42) {
		return 43;
	}
	else if (idx == 43) {
		return 44;
	}
	else if (idx == 44) {
		return 45;
	}
	else if (idx == 45) {
		return 46;
	}
	else if (idx == 46) {
		return 47;
	}
	else if (idx == 47) {
		return 48;
	}
	else if (idx == 48) {
		return 49;
	}
	else if (idx == 49) {
		return 50;
	}
	else if (idx == 50) {
		return 51;
	}
	else if (idx == 51) {
		return 52;
	}
	else if (idx == 52) {
		return 53;
	}
	else if (idx == 53) {
		return 54;
	}
	else if (idx == 54) {
		return 55;
	}
	else if (idx == 55) {
		return 56;
	}
	else if (idx == 56) {
		return 57;
	}
	else if (idx == 57) {
		return 58;
	}
	else if (idx == 58) {
		return 59;
	}
	else if (idx == 59) {
		return 60;
	}
	else if (idx == 60) {
		return 61;
	}
	else if (idx == 61) {
		return 62;
	}
	else if (idx == 62) {
		return 63;
	}
	else {
		return 0;
	}
}



/* --------------------- sifive_gpio0 ------------ */


/* --------------------- sifive_gpio_button ------------ */


/* --------------------- sifive_gpio_led ------------ */


/* --------------------- sifive_gpio_switch ------------ */


/* --------------------- sifive_i2c0 ------------ */


/* --------------------- sifive_pwm0 ------------ */


/* --------------------- sifive_remapper2 ------------ */


/* --------------------- sifive_rtc0 ------------ */



/* --------------------- sifive_test0 ------------ */
static __inline__ unsigned long __metal_driver_sifive_test0_base(const struct __metal_shutdown *sd)
{
	if ((uintptr_t)sd == (uintptr_t)&__metal_dt_teststatus_4000) {
		return METAL_SIFIVE_TEST0_4000_BASE_ADDRESS;
	}
	else {
		return 0;
	}
}

static __inline__ unsigned long __metal_driver_sifive_test0_size(const struct __metal_shutdown *sd)
{
	if ((uintptr_t)sd == (uintptr_t)&__metal_dt_teststatus_4000) {
		return METAL_SIFIVE_TEST0_4000_SIZE;
	}
	else {
		return 0;
	}
}



/* --------------------- sifive_trace ------------ */

/* --------------------- sifive_uart0 ------------ */


/* --------------------- sifive_simuart0 ------------ */


/* --------------------- sifive_wdog0 ------------ */


/* --------------------- sifive_fe310_g000_hfrosc ------------ */


/* --------------------- sifive_fe310_g000_hfxosc ------------ */


/* --------------------- sifive_fe310_g000_lfrosc ------------ */


/* --------------------- sifive_fe310_g000_pll ------------ */


/* --------------------- sifive_fe310_g000_prci ------------ */


#define __METAL_DT_MAX_MEMORIES 2

struct metal_memory *__metal_memory_table[] __attribute__((weak)) = {
					&__metal_dt_mem_testram_40000000,
					&__metal_dt_mem_memory_80000000};

/* From clint@2000000 */
#define __METAL_DT_RISCV_CLINT0_HANDLE (&__metal_dt_clint_2000000.controller)

#define __METAL_DT_CLINT_2000000_HANDLE (&__metal_dt_clint_2000000.controller)

#define __METAL_DT_MAX_HARTS 1

#define __METAL_CPU_0_ICACHE_HANDLE 1

#define __METAL_CPU_0_DCACHE_HANDLE 1

struct __metal_driver_cpu *__metal_cpu_table[] __attribute__((weak))  = {
					&__metal_dt_cpu_0};

/* From interrupt_controller@c000000 */
#define __METAL_DT_RISCV_PLIC0_HANDLE (&__metal_dt_interrupt_controller_c000000.controller)

#define __METAL_DT_INTERRUPT_CONTROLLER_C000000_HANDLE (&__metal_dt_interrupt_controller_c000000.controller)

#define __METAL_DT_PMP_HANDLE (&__metal_dt_pmp)

/* From global_external_interrupts */
#define __METAL_DT_SIFIVE_GLOBAL_EXINTR0_HANDLE (&__metal_dt_global_external_interrupts.irc)

#define __METAL_DT_GLOBAL_EXTERNAL_INTERRUPTS_HANDLE (&__metal_dt_global_external_interrupts.irc)

#define __MEE_DT_MAX_GPIOS 0

struct __metal_driver_sifive_gpio0 *__metal_gpio_table[] __attribute__((weak))  = {
					NULL };
#define __METAL_DT_MAX_BUTTONS 0

struct __metal_driver_sifive_gpio_button *__metal_button_table[] __attribute__((weak))  = {
					NULL };
#define __METAL_DT_MAX_LEDS 0

struct __metal_driver_sifive_gpio_led *__metal_led_table[] __attribute__((weak))  = {
					NULL };
#define __METAL_DT_MAX_SWITCHES 0

struct __metal_driver_sifive_gpio_switch *__metal_switch_table[] __attribute__((weak))  = {
					NULL };
#define __METAL_DT_MAX_I2CS 0

struct __metal_driver_sifive_i2c0 *__metal_i2c_table[] __attribute__((weak))  = {
					NULL };
#define __METAL_DT_MAX_PWMS 0

struct __metal_driver_sifive_pwm0 *__metal_pwm_table[] __attribute__((weak))  = {
					NULL };
#define __METAL_DT_MAX_RTCS 0

struct __metal_driver_sifive_rtc0 *__metal_rtc_table[] __attribute__((weak))  = {
					NULL };
#define __METAL_DT_MAX_SPIS 0

struct __metal_driver_sifive_spi0 *__metal_spi_table[] __attribute__((weak))  = {
					NULL };
/* From teststatus@4000 */
#define __METAL_DT_SHUTDOWN_HANDLE (&__metal_dt_teststatus_4000.shutdown)

#define __METAL_DT_TESTSTATUS_4000_HANDLE (&__metal_dt_teststatus_4000.shutdown)

#define __METAL_DT_MAX_UARTS 0

struct __metal_driver_sifive_uart0 *__metal_uart_table[] __attribute__((weak))  = {
					NULL };
#define __METAL_DT_MAX_SIMUARTS 0

struct __metal_driver_sifive_simuart0 *__metal_simuart_table[] __attribute__((weak))  = {
					NULL };
#define __METAL_DT_MAX_WDOGS 0

struct __metal_driver_sifive_wdog0 *__metal_wdog_table[] __attribute__((weak))  = {
					NULL };
#endif /* MACROS_ELSE_METAL_H*/

#endif /* ! __METAL_MACHINE_MACROS */

#endif /* ! ASSEMBLY */
