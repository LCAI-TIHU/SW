/*
 * aipu_timer.c
 *
 *  Created on: Nov 30, 2021
 *      Author: debug
 */

#include "aipu_timer_reg.h"
#include "aipu_io.h"
#include "metal/machine.h"
#include "dla_debug.h"
#include "aipu_timer.h"
#define TIMER_EN 						1<<0
#define TIMER_CASCADE					1<<3

static uint64_t clock_rate;

static __inline__ uint64_t __get_cpu_clock(struct aipu_timer *timer, struct metal_clock *system_clk)
{
	clock_rate = metal_clock_get_rate_hz(system_clk);
	return clock_rate;
}

static __inline__ int32_t __aipu_timer_init(struct aipu_timer *timer, struct metal_clock *system_clk)
{
	clock_rate = __get_cpu_clock(timer, system_clk);
	REG_RW(timer->timer_base, REGISTER_BANK_CLOCK_ENABLE_SET) = 0x00000001;
    while(REG_RW(timer->timer_base, REGISTER_BANK_CLOCK_ENABLE_STATUS) == 0);

    REG_RW(timer->timer_base, KERNEL_CLOCK_ENABLE_SET) = 0x00000001;
    while(REG_RW(timer->timer_base, KERNEL_CLOCK_ENABLE_STATUS) == 0);
    //casecade
    REG_RW(timer->timer_base, TIMER1CTRL) = (0 << 8) | (1 << 7) | (0 << 6) | (0 << 5) | (0 << 4) | (1 << 3) | (0 << 2) | (0 << 1) | (0 << 0);
    //timer_en
    REG_RW(timer->timer_base, TIMER2CTRL) = (0 << 8) | (0 << 7) | (1 << 6) | (0 << 5) | (0 << 4) | (0 << 3) | (0 << 2) | (0 << 1) | (1 << 0);
    return 0;
}

static __inline__ void __aipu_timer_1_prescale_set(struct aipu_timer *timer, uint32_t prescale)
{
	REG_RW(timer->timer_base, TIMER1CTRL) &= ~(TIMER_EN);

	if(prescale == 1)
		REG_RW(timer->timer_base, TIMER1CTRL) &= 0xffffffcf;
	else if(prescale == 16){
		REG_RW(timer->timer_base, TIMER1CTRL) &= 0xffffffdf;
		REG_RW(timer->timer_base, TIMER1CTRL) |= 0x00000010;
	}
	else if(prescale == 256){
		REG_RW(timer->timer_base, TIMER2CTRL) &= 0xffffffef;
		REG_RW(timer->timer_base, TIMER2CTRL) |= 0x00000020;
	}
	else
		REG_RW(timer->timer_base, TIMER1CTRL) &= 0xffffffcf;
//	REG_RW(timer->timer_base, TIMER1CTRL) |= TIMER_EN;
}
static __inline__ void __aipu_timer_2_prescale_set(struct aipu_timer *timer, uint32_t prescale)
{
	REG_RW(timer->timer_base, TIMER2CTRL) &= ~(TIMER_EN);

	if(prescale == 1)
		REG_RW(timer->timer_base, TIMER2CTRL) &= 0xffffffe7;
	else if(prescale == 16){
		REG_RW(timer->timer_base, TIMER2CTRL) &= 0xffffffef;
		REG_RW(timer->timer_base, TIMER2CTRL) |= 0x00000008;
	}
	else if(prescale == 256){
		REG_RW(timer->timer_base, TIMER2CTRL) &= 0xfffffff7;
		REG_RW(timer->timer_base, TIMER2CTRL) |= 0x00000010;
	}
	else
		REG_RW(timer->timer_base, TIMER2CTRL) &= 0xffffffe7;
//	REG_RW(timer->timer_base, TIMER2CTRL) |= TIMER_EN;
}
static __inline__ uint32_t __aipu_timer_1_prescale_get(struct aipu_timer *timer)
{
	uint32_t prescale;
	if((REG_RW(timer->timer_base, TIMER1CTRL) & 0x00000030) == 0x00000000)
		prescale = 1;
	else if((REG_RW(timer->timer_base, TIMER1CTRL) & 0x00000030) == 0x00000010)
		prescale = 16;
	else if((REG_RW(timer->timer_base, TIMER1CTRL) & 0x00000030) == 0x00000020)
		prescale = 256;
	else
		prescale = 1;
	return prescale;
}
static __inline__ uint32_t __aipu_timer_2_prescale_get(struct aipu_timer *timer)
{
	uint32_t prescale;
	if((REG_RW(timer->timer_base, TIMER2CTRL) & 0x000000018) == 0x00000000)
		prescale = 1;
	else if((REG_RW(timer->timer_base, TIMER2CTRL) & 0x000000018) == 0x00000008)
		prescale = 16;
	else if((REG_RW(timer->timer_base, TIMER2CTRL) & 0x000000018) == 0x00000010)
		prescale = 256;
	else
		prescale = 1;
	return prescale;
}
static __inline__ uint64_t __get_timer_ticks(struct aipu_timer *timer)
{
    uint64_t low_s, hi_s;
    uint64_t timer_val;
	uint32_t prescale1 = __aipu_timer_1_prescale_get(timer);
	uint32_t prescale2 = __aipu_timer_2_prescale_get(timer);
	REG_RW(timer->timer_base, TIMER1READREQ) = 0x00000001;
	REG_RW(timer->timer_base, TIMER2READREQ) = 0x00000001;
	if((REG_RW(timer->timer_base, TIMER1CTRL) & TIMER_CASCADE) == TIMER_CASCADE)
	{
		/* get current timer value */
		low_s = REG_RW(timer->timer_base, TIMER1CURVALUE)*prescale1;
		hi_s = REG_RW(timer->timer_base, TIMER2CURVALUE)*prescale2;
	}else{
		low_s = REG_RW(timer->timer_base, TIMER1CURVALUE)*prescale1;
		hi_s = 0;
	}
	timer_val = (((hi_s) << 32) + low_s);
    return timer_val;
}


static __inline__ uint64_t __get_timer_us(struct aipu_timer *timer)
{
	return __get_timer_ticks(timer) / (clock_rate / 1000000);
}

static __inline__ void __set_timer_udelay(struct aipu_timer *timer, uint32_t usec)
{
    uint64_t end;
    end = __get_timer_us(timer) - (uint64_t)usec * clock_rate / 1000000;
    while(__get_timer_us(timer) > end);
}
static __inline__ void __aipu_timer_1_enable(struct aipu_timer *timer)
{
	REG_RW(timer->timer_base, TIMER1CTRL) |= TIMER_EN;
}
static __inline__ void __aipu_timer_1_disable(struct aipu_timer *timer)
{
	REG_RW(timer->timer_base, TIMER1CTRL) &= ~(TIMER_EN);
}
static __inline__ void __aipu_timer_1_halt(struct aipu_timer *timer)
{
	REG_RW(timer->timer_base, TIMER1CTRL) |= 1<<8;
}
static __inline__ void __aipu_timer_1_continue(struct aipu_timer *timer)
{
	REG_RW(timer->timer_base, TIMER1CTRL) &= ~(1<<8);
}
static __inline__ uint32_t __aipu_timer_1_status(struct aipu_timer *timer)
{
	return REG_RW(timer->timer_base, TIMER1CTRL);
}
static __inline__ void __aipu_timer_1_loadvalue_set(struct aipu_timer *timer, uint32_t loadvalue)
{
	REG_RW(timer->timer_base, TIMER1LOAD) = loadvalue;
}
static __inline__ void __aipu_timer_2_enable(struct aipu_timer *timer)
{
	REG_RW(timer->timer_base, TIMER2CTRL) |= TIMER_EN;
}
static __inline__ void __aipu_timer_2_disable(struct aipu_timer *timer)
{
	REG_RW(timer->timer_base, TIMER2CTRL) &= ~(TIMER_EN);
}
static __inline__ void __aipu_timer_2_halt(struct aipu_timer *timer)
{
	REG_RW(timer->timer_base, TIMER2CTRL) |= 1<<7;// not 1<<8
}
static __inline__ void __aipu_timer_2_continue(struct aipu_timer *timer)
{
	REG_RW(timer->timer_base, TIMER2CTRL) &= ~(1<<7);
}
static __inline__ uint32_t __aipu_timer_2_status(struct aipu_timer *timer)
{
	return REG_RW(timer->timer_base, TIMER2CTRL);
}
static __inline__ void __aipu_timer_2_loadvalue_set(struct aipu_timer *timer, uint32_t loadvalue)
{
	REG_RW(timer->timer_base, TIMER2LOAD) = loadvalue;
}
static __inline__ void __aipu_timer_loadvalue_set(struct aipu_timer *timer, uint32_t loadvalue1, uint32_t loadvalue2)
{	//must be cascade
	REG_RW(timer->timer_base, TIMER1CTRL) &= ~(TIMER_EN);
	REG_RW(timer->timer_base, TIMER2CTRL) &= ~(TIMER_EN);
	REG_RW(timer->timer_base, TIMER1LOAD) = loadvalue1;
	REG_RW(timer->timer_base, TIMER2LOAD) = loadvalue2;
	REG_RW(timer->timer_base, TIMER2CTRL) |= TIMER_EN;
	REG_RW(timer->timer_base, TIMER1CTRL) |= TIMER_EN;
}
__METAL_DEFINE_VTABLE(aipu_timer_vtable) = {
		.get_cpu_clock = &__get_cpu_clock,
		.aipu_timer_init = &__aipu_timer_init,
		.get_timer_ticks = &__get_timer_ticks,
		.get_timer_us = &__get_timer_us,
		.set_timer_udelay = &__set_timer_udelay,
		.aipu_timer_1_enable = &__aipu_timer_1_enable,
		.aipu_timer_1_disable = &__aipu_timer_1_disable,
		.aipu_timer_1_halt = &__aipu_timer_1_halt,
		.aipu_timer_1_continue = &__aipu_timer_1_continue,
		.aipu_timer_1_prescale_set = &__aipu_timer_1_prescale_set,
		.aipu_timer_1_prescale_get = &__aipu_timer_1_prescale_get,
		.aipu_timer_1_loadvalue_set = &__aipu_timer_1_loadvalue_set,
		.aipu_timer_1_status = &__aipu_timer_1_status,
		.aipu_timer_2_enable = &__aipu_timer_2_enable,
		.aipu_timer_2_disable = &__aipu_timer_2_disable,
		.aipu_timer_2_halt = &__aipu_timer_2_halt,
		.aipu_timer_2_continue = &__aipu_timer_2_continue,
		.aipu_timer_2_status = &__aipu_timer_2_status,
		.aipu_timer_2_prescale_set = &__aipu_timer_2_prescale_set,
		.aipu_timer_2_prescale_get = &__aipu_timer_2_prescale_get,
		.aipu_timer_2_loadvalue_set = &__aipu_timer_2_loadvalue_set,
		.aipu_timer_loadvalue_set = &__aipu_timer_loadvalue_set,
};

