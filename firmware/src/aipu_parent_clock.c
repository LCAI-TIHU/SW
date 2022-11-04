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
