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
