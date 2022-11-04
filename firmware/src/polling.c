#include <aipu_io.h>
#include <polling.h>
int32_t polling_expect_by_count(void * addr, uint32_t expect, uint32_t mask, uint32_t count)
{
    uint32_t i = 0;
    uint32_t val;

    do
    {
        i++;
        val = read32(addr);

        if ((val & mask) == (expect & mask))
            return 0;

    } while(i < count);

    //FIXME: show warning here

    return -1;
}

int32_t polling_unexpect_by_count(void * addr, uint32_t unexpect, uint32_t mask, uint32_t count)
{
    uint32_t i = 0;
    uint32_t val;

    do
    {
        i++;
        val = read32(addr);

        if ((val & mask) != (unexpect & mask))
            return 0;

    } while(i < count);

    //FIXME: show warning here

    return -1;
}
