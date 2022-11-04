#ifndef POLLING_H
#define POLLING_H
#include <stdint.h>
int32_t polling_expect_by_count(void * addr, uint32_t expect, uint32_t mask, uint32_t count);
int32_t polling_unexpect_by_count(void * addr, uint32_t unexpect, uint32_t mask, uint32_t count);

#endif
