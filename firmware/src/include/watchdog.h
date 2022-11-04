#ifndef _WATCHDOG_H_
#define _WATCHDOG_H_

#include <stdint.h>
#include <aipu_io.h>

#define WDT_BASE_ADDR                                   0X40300000
#define WDT_CONTROL_OFFSET                              0x00000c40
#define WDT_LOAD_VALUE_OFFSET                           0x00000c44
#define WDT_INT_LEVEL_OFFSET                            0x00000c48
#define WDT_RLD_UPD_OFFSET                              0x00000c4c
#define WDT_CNT_VLU_OFFSET                              0x00000c50
#define WDT_REG_RW(offset)              REG_RW(WDT_BASE_ADDR,offset)

extern  uint32_t __inline__ wdt_enable(void)
{
        WDT_REG_RW(WDT_CONTROL_OFFSET) |= 0x00000001;
        return WDT_REG_RW(WDT_CONTROL_OFFSET);
}
extern  uint32_t __inline__ wdt_disable(void)
{
        WDT_REG_RW(WDT_CONTROL_OFFSET) &= 0xfffffffe;
        return WDT_REG_RW(WDT_CONTROL_OFFSET);
}

extern uint32_t __inline__ wdt_load_value_set(uint32_t val)
{
        WDT_REG_RW(WDT_LOAD_VALUE_OFFSET) = val;
        return WDT_REG_RW(WDT_LOAD_VALUE_OFFSET);
}
extern uint32_t __inline__ wdt_set_timer_resolution_2320(void)
{
        WDT_REG_RW(WDT_CONTROL_OFFSET) &= 0xfffffffd;
        return WDT_REG_RW(WDT_CONTROL_OFFSET);
}
extern uint32_t __inline__ wdt_set_timer_resolution_260000(void)
{
        WDT_REG_RW(WDT_CONTROL_OFFSET) |= 0x00000002;
        return WDT_REG_RW(WDT_CONTROL_OFFSET);
}
extern void __inline__ wdt_load_update(void)
{
        WDT_REG_RW(WDT_RLD_UPD_OFFSET) = 0x00000001;
}
extern uint32_t __inline__ wdt_int_level_set(uint32_t val)
{
        WDT_REG_RW(WDT_INT_LEVEL_OFFSET) = val;
        return WDT_REG_RW(WDT_INT_LEVEL_OFFSET);
}
extern uint32_t __inline__ wdt_feed(uint32_t val)
{
		WDT_REG_RW(WDT_LOAD_VALUE_OFFSET) = val;
        return WDT_REG_RW(WDT_LOAD_VALUE_OFFSET);
}
extern int32_t __inline__ wdt_init(uint32_t val)
{
		int32_t rc;
		WDT_REG_RW(WDT_LOAD_VALUE_OFFSET) = val;
		WDT_REG_RW(WDT_CONTROL_OFFSET) |= 0x00000003;
		if(WDT_REG_RW(WDT_LOAD_VALUE_OFFSET) != val)
			return 1;
		if((WDT_REG_RW(WDT_CONTROL_OFFSET)&0x00000003) != 0x00000003)
			return 1;
		return 0;
}

#endif /* _WATCHDOG_H_ */
