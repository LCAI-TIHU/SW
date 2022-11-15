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

#include <metal/machine/platform.h>
#include <aipu_parent_clock.h>
#include <metal/machine.h>
#include <stddef.h>

static long aipu_clock = 25000000;

long __metal_driver_aipu_clock_get_rate_hz(const struct metal_clock *gclk) {
    return aipu_clock;
}

long __metal_driver_aipu_clock_set_rate_hz(struct metal_clock *gclk,
                                            long target_hz) {
	aipu_clock = target_hz;
    return aipu_clock;
}

__METAL_DEFINE_VTABLE(__metal_driver_vtable_aipu_clock) = {
    .clock.get_rate_hz = &__metal_driver_aipu_clock_get_rate_hz,
    .clock.set_rate_hz = &__metal_driver_aipu_clock_set_rate_hz,
};
