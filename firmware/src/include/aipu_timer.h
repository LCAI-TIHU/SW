/* Copyright 2019 Inspur Corporation. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
    
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef AIPU_TIMER_H
#define AIPU_TIMER_H
#include <metal/compiler.h>
#include "metal/clock.h"
#include <stdint.h>
#define TIMER0_BASE 0x40110000

struct aipu_timer;

struct aipu_timer_vtable{
	uint64_t (*get_cpu_clock)(struct aipu_timer *timer, struct metal_clock *system_clk);
	int32_t (*aipu_timer_init)(struct aipu_timer *timer, struct metal_clock *system_clk);
	uint64_t (*get_timer_ticks)(struct aipu_timer *timer);
	uint64_t (*get_timer_us)(struct aipu_timer *timer);
	void (*set_timer_udelay)(struct aipu_timer *timer, uint32_t usec);
	void (*aipu_timer_1_enable)(struct aipu_timer *timer);
	void (*aipu_timer_1_disable)(struct aipu_timer *timer);
	void (*aipu_timer_1_halt)(struct aipu_timer *timer);
	void (*aipu_timer_1_continue)(struct aipu_timer *timer);
	uint32_t (*aipu_timer_1_status)(struct aipu_timer *timer);
	void (*aipu_timer_1_prescale_set)(struct aipu_timer *timer, uint32_t prescale);
	uint32_t (*aipu_timer_1_prescale_get)(struct aipu_timer *timer);
	void (*aipu_timer_1_loadvalue_set)(struct aipu_timer *timer, uint32_t loadvalue);
	void (*aipu_timer_2_enable)(struct aipu_timer *timer);
	void (*aipu_timer_2_disable)(struct aipu_timer *timer);
	void (*aipu_timer_2_halt)(struct aipu_timer *timer);
	void (*aipu_timer_2_continue)(struct aipu_timer *timer);
	uint32_t (*aipu_timer_2_status)(struct aipu_timer *timer);
	void (*aipu_timer_2_prescale_set)(struct aipu_timer *timer, uint32_t prescale);
	uint32_t (*aipu_timer_2_prescale_get)(struct aipu_timer *timer);
	void (*aipu_timer_2_loadvalue_set)(struct aipu_timer *timer, uint32_t loadvalue);
	void (*aipu_timer_loadvalue_set)(struct aipu_timer *timer, uint32_t loadvalue1, uint32_t loadvalue2);
};

struct aipu_timer{
	const struct aipu_timer_vtable *vtable;
	uint64_t timer_base;
};
__METAL_DECLARE_VTABLE(aipu_timer_vtable);

static __inline__ uint64_t get_cpu_clock(struct aipu_timer *timer, struct metal_clock *system_clk){
	return timer->vtable->get_cpu_clock(timer, system_clk);
}
static __inline__ int32_t aipu_timer_init(struct aipu_timer *timer, struct metal_clock *system_clk){
	return timer->vtable->aipu_timer_init(timer, system_clk);
}
static __inline__ uint64_t get_timer_ticks(struct aipu_timer *timer){
	return timer->vtable->get_timer_ticks(timer);
}
static __inline__ uint64_t get_timer_us(struct aipu_timer *timer){
	return timer->vtable->get_timer_us(timer);
}
static __inline__ void set_timer_udelay(struct aipu_timer *timer, uint32_t usec){
	timer->vtable->set_timer_udelay(timer, usec);
}
static __inline__ void aipu_timer_1_enable(struct aipu_timer *timer){
	timer->vtable->aipu_timer_1_enable(timer);
}
static __inline__ void aipu_timer_1_disable(struct aipu_timer *timer){
	timer->vtable->aipu_timer_1_disable(timer);
}
static __inline__ void aipu_timer_1_halt(struct aipu_timer *timer){
	timer->vtable->aipu_timer_1_halt(timer);
}
static __inline__ void aipu_timer_1_continue(struct aipu_timer *timer){
	timer->vtable->aipu_timer_1_continue(timer);
}
static __inline__ uint32_t aipu_timer_1_status(struct aipu_timer *timer){
	return timer->vtable->aipu_timer_1_status(timer);
}
static __inline__ void aipu_timer_1_prescale_set(struct aipu_timer *timer, uint32_t prescale){
	timer->vtable->aipu_timer_1_prescale_set(timer, prescale);
}
static __inline__ uint32_t aipu_timer_1_prescale_get(struct aipu_timer *timer){
	return timer->vtable->aipu_timer_1_prescale_get(timer);
}
static __inline__ void aipu_timer_1_loadvalue_set(struct aipu_timer *timer, uint32_t loadvalue){
	timer->vtable->aipu_timer_1_loadvalue_set(timer, loadvalue);
}

static __inline__ void aipu_timer_2_enable(struct aipu_timer *timer){
	timer->vtable->aipu_timer_2_enable(timer);
}
static __inline__ void aipu_timer_2_disable(struct aipu_timer *timer){
	timer->vtable->aipu_timer_2_disable(timer);
}
static __inline__ void aipu_timer_2_halt(struct aipu_timer *timer){
	timer->vtable->aipu_timer_2_halt(timer);
}
static __inline__ void aipu_timer_2_continue(struct aipu_timer *timer){
	timer->vtable->aipu_timer_2_continue(timer);
}
static __inline__ uint32_t aipu_timer_2_status(struct aipu_timer *timer){
	return timer->vtable->aipu_timer_2_status(timer);
}
static __inline__ void aipu_timer_2_prescale_set(struct aipu_timer *timer, uint32_t prescale){
	timer->vtable->aipu_timer_2_prescale_set(timer, prescale);
}
static __inline__ uint32_t aipu_timer_2_prescale_get(struct aipu_timer *timer){
	return timer->vtable->aipu_timer_2_prescale_get(timer);
}
static __inline__ void aipu_timer_2_loadvalue_set(struct aipu_timer *timer, uint32_t loadvalue){
	timer->vtable->aipu_timer_2_loadvalue_set(timer, loadvalue);
}
static __inline__ void aipu_timer_loadvalue_set(struct aipu_timer *timer, uint32_t loadvalue1, uint32_t loadvalue2){
	timer->vtable->aipu_timer_loadvalue_set(timer, loadvalue1, loadvalue2);
}
#endif
