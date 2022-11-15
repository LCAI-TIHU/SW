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

#include <stdint.h>
//#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "device_init.h"
#include "cpu_callback.h"
//#include "dla_debug.h"
#include "operations.h"

extern struct cpu_device cpu_dev;

int8_t getBpe(uint32_t format)
{
    int8_t bpe = -1;
    switch (format)
    {
    case RFLOAT:
    case RINT:
    case RUINT:
        bpe = 4;
        break;
    case RBFLOAT:
        bpe = 2;
        break;
    case RINT8:
        bpe = 1;
        break;
    default:
        bpe = -1;
    }
    return bpe;
}
uint32_t getAddrOffset(uint32_t input_datatype, int32_t line_stride, int32_t surf_stride, uint32_t h, uint32_t w, uint32_t c, uint32_t *offset)
{
    uint32_t err = 0;

    uint32_t x = 0;
    uint32_t xStride = 0;
    uint32_t cquotient = 0;
    uint32_t cremainder = 0;

    int8_t bpe = getBpe(input_datatype);
    if (bpe < 0)
    {
        return err = EINVAL;
    }

    switch (input_datatype)
    {
    case RFLOAT:
    case RBFLOAT:
    case RINT:
    case RUINT:
    case RINT8:
        x = 32 / bpe;
        xStride = x * bpe;
        cquotient = c / x;
        cremainder = c % x;
        *offset = (cquotient * (surf_stride)) + (h * (line_stride)) + (w * xStride) + (cremainder * bpe);
        break;
    default:
        *offset = 0;
        debug_info("Error, Unsupported input format: %d\n", input_datatype);
        err = EINVAL;
    }

fail:
    return err;
}
// chang h->w->c to surf->h->w->c, and add line_stride and surf_stride, attention after vectorize, hwc does not change
int32_t vectorize(void *dat_in, void *dat_out, struct op_buffer_desc *dst_data)
{
    /* debug_info("\nEnter %s\n", __func__); */
    uint32_t dat_h = dst_data->height;
    uint32_t dat_w = dst_data->width;
    uint32_t dat_c = dst_data->channel;
    int32_t line_stride_out = dst_data->line_stride;
    int32_t surf_stride_out = dst_data->surf_stride;
    uint32_t c, h, w, offset;
    uint32_t dat_type = dst_data->datatype;
    if (dat_type != RINT8)
    {
        debug_info("Error, unsupport data type!\n");
        return TYPE_ERR;
    }
    memset(dat_out, 0, ((uint32_t)((dat_c - 1) / C_ATM + 1)) * surf_stride_out);
    for (h = 0; h < dat_h; h++)
    {
        /* debug_info("\nh = %d\n", h); */
        for (w = 0; w < dat_w; w++)
        {
            /* debug_info("\nw = %d\n", w); */
            for (c = 0; c < dat_c; c++)
            {
                getAddrOffset(dat_type, line_stride_out, surf_stride_out, h, w, c, &offset);
                int8_t *input_addr = (int8_t *)((uint64_t)dat_in + c + w * dat_c + h * dat_w * dat_c);
                int8_t data = *(volatile int8_t *)input_addr;
                int8_t *output_addr = (int8_t *)((uint64_t)dat_out + offset);
                *(volatile int8_t *)output_addr = data;
                /* debug_info("%d  ", *(volatile int8_t *)output_addr); */
            }
        }
    }
    // debug_info("\nExit %s!\n", __func__);
    return 0;
}
// change surf->h->w->c to h->w->c, and remove line _stride and surf_stride. attention after reshape, hwc does not change.
uint32_t unvectorize(void *dat_in, void *dat_out, struct op_buffer_desc *src_data)
{
    /* debug_info("\nEnter %s\n", __func__); */
    uint32_t dat_h = src_data->height;
    uint32_t dat_w = src_data->width;
    uint32_t dat_c = src_data->channel;
    int32_t line_stride_in = src_data->line_stride;
    int32_t surf_stride_in = src_data->surf_stride;
    uint32_t dat_type = src_data->datatype;
    uint32_t c, h, w, offset;
    if (dat_type != RINT8)
    {
        debug_info("Unsupport data type!\n");
        return TYPE_ERR;
    }
    for (h = 0; h < dat_h; h++)
    {
        /* debug_info("\nh = %d\n", h); */
        for (w = 0; w < dat_w; w++)
        {
            /* debug_info("\nw = %d\n", w); */
            for (c = 0; c < dat_c; c++)
            {
                getAddrOffset(dat_type, line_stride_in, surf_stride_in, h, w, c, &offset);
                int8_t *input_addr = (int8_t *)((uint64_t)dat_in + offset);
                int8_t data = *(volatile int8_t *)input_addr;
                int8_t *output_addr = (int8_t *)((uint64_t)dat_out + c + w * dat_c + h * dat_w * dat_c);
                *(volatile int8_t *)output_addr = data;
                /* debug_info("%d  ", *(volatile int8_t *)output_addr); */
            }
        }
    }
    /* debug_info("\nExit %s!\n", __func__); */
    return 0;
}
#ifndef __MATH_VECTOR

int32_t executeSoftmax(struct cpu_param cpu_parameters)
{
    // debug_info("Enter %s\n", __func__);
    int32_t err = 0;
    int32_t h, w, c;

    uint64_t output_addr_temp, input_addr_temp;

    uint64_t pDst = cpu_dev.task->address_list[cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.index];
    uint32_t dat_h_out = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.height;
    uint32_t dat_w_out = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.width;
    uint32_t dat_c_out = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.channel;
    int32_t line_stride_out = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.line_stride;
    int32_t surf_stride_out = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.surf_stride;
    uint32_t dat_type_out = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.datatype;
    uint32_t dat_type_out_size = (dat_type_out == RINT8) ? 1 : (dat_type_out == RBFLOAT) ? 2
                                                                                         : 4;
    uint32_t size_out = dat_h_out * dat_w_out * dat_c_out;

    uint64_t pSrc = cpu_dev.task->address_list[cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.index];
    uint32_t dat_type_in = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.datatype;
    uint32_t dat_type_in_size = (dat_type_in == RINT8) ? 1 : (dat_type_in == RBFLOAT) ? 2
                                                                                      : 4;

    uint32_t dat_h_in = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.height;
    uint32_t dat_w_in = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.width;
    uint32_t dat_c_in = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.channel;
    int32_t line_stride_in = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.line_stride;
    int32_t surf_stride_in = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.surf_stride;
    uint32_t size_in = dat_h_in * dat_w_in * dat_c_in;

    if ((line_stride_in > 0) || (surf_stride_in > 0))
    {
        if (dat_type_in != RINT8)
        {
            debug_info("%s Wrong datatype!\n", __func__);
            return TYPE_ERR;
        }
        void *temp0 = (void *)malloc(size_in * dat_type_in_size);
        err = unvectorize((void *)pSrc, (void *)temp0, &cpu_parameters.cpu_operation_buffer.reduce_buffers.src_data);
        input_addr_temp = (uint64_t)malloc(size_in * sizeof(float));
        for (h = 0; h < dat_h_in; h++)
        {
            for (w = 0; w < dat_w_in; w++)
            {
                for (c = 0; c < dat_c_in; c++)
                {
                    *(volatile float *)(input_addr_temp + (c + w * dat_c_in + h * dat_w_in * dat_c_in) * sizeof(float)) =
                        *(volatile int8_t *)((uint64_t)temp0 + (c + w * dat_c_in + h * dat_w_in * dat_c_in) * sizeof(int8_t)) *
                        cpu_parameters.cpu_operation.reduce_op.common.input_scale_factor[0];
                }
            }
        }
        free(temp0);
    }
    else
    {
        input_addr_temp = pSrc;
    }

    if ((line_stride_out > 0) || (surf_stride_out > 0))
    {
        if (dat_type_out != RINT8)
            return TYPE_ERR;
        output_addr_temp = (uint64_t)malloc(size_out * dat_type_out_size);
    }
    else
    {
        output_addr_temp = pDst;
    }

    float_t maxval[dat_h_in * dat_w_in];
    /* debug_info("\nInput shape is %d x %d x %d.\n", dat_h_in, dat_w_in, dat_c_in); */
    for (h = 0; h < dat_h_in; h++)
    {
        for (w = 0; w < dat_w_in; w++)
        {
            maxval[w + h * dat_w_in] = -INFINITY;
            for (c = 0; c < dat_c_in; c++)
            {
                uint64_t input_addr = input_addr_temp + (c + w * dat_c_in + h * dat_w_in * dat_c_in) * sizeof(float);
                float input_data = *(volatile float *)input_addr;
                if (input_data > maxval[w + h * dat_w_in])
                {
                    maxval[w + h * dat_w_in] = input_data;
                }
            }
        }
    }
    /* debug_info("\nMaxval is : "); */
    /* print_float(maxval[0]); debug_info("\n"); */

    float_t sumexp[dat_h_in * dat_w_in];
    for (h = 0; h < dat_h_in; h++)
    {
        for (w = 0; w < dat_w_in; w++)
        {
            sumexp[w + h * dat_w_in] = 0.0f;
            for (c = 0; c < dat_c_in; c++)
            {
                uint64_t input_addr = input_addr_temp + (c + w * dat_c_in + h * dat_w_in * dat_c_in) * sizeof(float);
                float input_data = *(volatile float *)input_addr;
                /* print_float(input_data); debug_info(" -> "); */
                float exp_tmp = expf(input_data - maxval[w + h * dat_w_in]);
                /* print_float(exp_tmp); debug_info(" \n "); */
                sumexp[w + h * dat_w_in] += exp_tmp;
                *(volatile float *)input_addr = exp_tmp;
            }
        }
    }

    /* debug_info("\nSumexp is : "); */
    /* print_float(sumexp[0]); debug_info("\n"); */

    for (h = 0; h < dat_h_in; h++)
    {
        for (w = 0; w < dat_w_in; w++)
        {
            for (c = 0; c < dat_c_in; c++)
            {
                uint64_t input_addr = input_addr_temp + (c + w * dat_c_in + h * dat_w_in * dat_c_in) * sizeof(float);
                float input_data = *(volatile float *)input_addr;
                uint64_t output_addr = output_addr_temp + (c + w * dat_c_in + h * dat_w_in * dat_c_in) * dat_type_out_size;
                if (dat_type_out == RINT8)
                {
                    float y = input_data / sumexp[w + h * dat_w_in];
                    int8_t dstint8p = (y >= 127.0) ? 127 : (y >= -128.0) ? (int8_t)y
                                                                         : -128;
                    *(volatile int8_t *)output_addr = dstint8p;
                }
                else if (dat_type_out == RFLOAT)
                {
                    *(volatile float *)output_addr = input_data / sumexp[w + h * dat_w_in];
                }
                else
                {
                    debug_info("Error in %s : %d, unsupport output data type!\n");
                }
            }
        }
    }

    if ((line_stride_out > 0) || (surf_stride_out > 0))
    {
        err = vectorize((void *)output_addr_temp, (void *)pDst, &cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data);
        free((void *)output_addr_temp);
    }

    if ((line_stride_in > 0) || (surf_stride_in > 0))
    {
        free((void *)input_addr_temp);
    }
    cpu_output_debug(cpu_parameters);
    // debug_info("\nExit %s;\n", __func__);
    return err;
}
#endif

int32_t executeExpf_Tanh(struct cpu_param cpu_parameters)
{
    // debug_info("Enter %s\n", __func__);
    int32_t err = 0;
    int32_t h, w, c;

    uint64_t output_addr_temp, input_addr_temp;
    uint64_t pDst = cpu_dev.task->address_list[cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.index];
    uint32_t dat_h_out = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.height;
    uint32_t dat_w_out = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.width;
    uint32_t dat_c_out = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.channel;
    int32_t line_stride_out = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.line_stride;
    int32_t surf_stride_out = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.surf_stride;
    uint32_t dat_type_out = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.datatype;
    uint32_t dat_type_in = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.datatype;
    uint32_t dat_type_out_size = (dat_type_out == RINT8) ? 1 : (dat_type_out == RBFLOAT) ? 2
                                                                                         : 4;
    uint32_t size_out = dat_h_out * dat_w_out * dat_c_out;
    if ((line_stride_out > 0) || (surf_stride_out > 0))
    {
        if (dat_type_out != RINT8)
            return TYPE_ERR;
        output_addr_temp = (uint64_t)malloc(size_out * dat_type_out_size);
    }
    else
    {
        output_addr_temp = pDst;
    }
    uint64_t pSrc = cpu_dev.task->address_list[cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.index];
    uint32_t dat_type_in_size = (dat_type_in == RINT8) ? 1 : (dat_type_in == RBFLOAT) ? 2
                                                                                      : 4;
    uint32_t dat_h_in = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.height;
    uint32_t dat_w_in = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.width;
    uint32_t dat_c_in = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.channel;
    int32_t line_stride_in = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.line_stride;
    int32_t surf_stride_in = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.surf_stride;
    uint32_t size_in = dat_h_in * dat_w_in * dat_c_in;

    if (size_in != size_out)
    {
        debug_info("\nInput shape or output shape must be wrong!!!\n");
        return SHAPE_ERR;
    }

    if ((line_stride_in > 0) || (surf_stride_in > 0))
    {
        input_addr_temp = (uint64_t)malloc(size_in * dat_type_in_size);
        if (dat_type_in != RINT8)
        {
            debug_info("%s Wrong datatype!\n", __func__);
            return TYPE_ERR;
        }
        err = unvectorize((void *)pSrc, (void *)input_addr_temp, &cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data);
    }
    else
    {
        input_addr_temp = pSrc;
    }
    for (h = 0; h < dat_h_in; h++)
    {
        for (w = 0; w < dat_w_in; w++)
        {
            for (c = 0; c < dat_c_in; c++)
            {
                float_t y = 0.0;
                uint64_t input_addr = input_addr_temp + (c + w * dat_c_in + h * dat_w_in * dat_c_in) * dat_type_in_size;
                uint64_t output_addr = output_addr_temp + (c + w * dat_c_out + h * dat_w_out * dat_c_out) * dat_type_out_size;
                if ((dat_type_in == RINT8) && (dat_type_out == RFLOAT))
                {
                    float_t x = *(volatile int8_t *)input_addr * cpu_parameters.cpu_operation.common_only_op.common.input_scale_factor[0];
                    if (cpu_parameters.cpu_operation.with_weight_op.common.op_type == EXP)
                        y = expf(x);
                    else if (cpu_parameters.cpu_operation.with_weight_op.common.op_type == TANH)
                        y = tanh(x);
                    *(volatile float_t *)output_addr = y;
                }
                else if ((dat_type_in == RINT8) && (dat_type_out == RINT8))
                {
                    float_t x = *(volatile int8_t *)input_addr * cpu_parameters.cpu_operation.common_only_op.common.input_scale_factor[0];
                    if (cpu_parameters.cpu_operation.with_weight_op.common.op_type == EXP)
                        y = expf(x);
                    else if (cpu_parameters.cpu_operation.with_weight_op.common.op_type == TANH)
                        y = tanh(x);
                    y /= cpu_parameters.cpu_operation.common_only_op.common.output_scale_factor[0];
                    int8_t dstint8p = (y >= 127.0) ? 127 : (y >= -128.0) ? (int8_t)y
                                                                         : -128;
                    *(volatile int8_t *)output_addr = dstint8p;
                }
                else if ((dat_type_in == RFLOAT) && (dat_type_out == RFLOAT))
                {
                    float_t x = *(volatile float_t *)input_addr;
                    if (cpu_parameters.cpu_operation.with_weight_op.common.op_type == EXP)
                        y = expf(x);
                    else if (cpu_parameters.cpu_operation.with_weight_op.common.op_type == TANH)
                        y = tanh(x);
                    *(volatile float_t *)output_addr = y;
                }
                else if ((dat_type_in == RFLOAT) && (dat_type_out == RINT8))
                {
                    float_t x = *(volatile float_t *)input_addr;
                    if (cpu_parameters.cpu_operation.with_weight_op.common.op_type == EXP)
                        y = expf(x);
                    else if (cpu_parameters.cpu_operation.with_weight_op.common.op_type == TANH)
                        y = tanh(x);
                    y /= cpu_parameters.cpu_operation.common_only_op.common.output_scale_factor[0];
                    int8_t dstint8p = (y >= 127.0) ? 127 : (y >= -128.0) ? (int8_t)y
                                                                         : -128;
                    *(volatile int8_t *)output_addr = dstint8p;
                }
            }
        }
    }
    if ((line_stride_out > 0) || (surf_stride_out > 0))
    {
        err = vectorize((void *)output_addr_temp, (void *)pDst, &cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data);
        free((void *)output_addr_temp);
    }
    if ((line_stride_in > 0) || (surf_stride_in > 0))
    {
        free((void *)input_addr_temp);
    }
    // debug_info("\nExit %s;\n", __func__);
    // sifive_pl2cache0_flush((uintptr_t)pDst);
    cpu_output_debug(cpu_parameters);
    return err;
}

int32_t executeSigmoid(struct cpu_param cpu_parameters)
{
    // debug_info("Enter %s\n", __func__);
    int32_t err = 0;
    int32_t h, w, c;

    uint64_t output_addr_temp, input_addr_temp;
    uint64_t pDst = cpu_dev.task->address_list[cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.index];
    uint32_t dat_h_out = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.height;
    uint32_t dat_w_out = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.width;
    uint32_t dat_c_out = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.channel;
    int32_t line_stride_out = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.line_stride;
    int32_t surf_stride_out = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.surf_stride;
    uint32_t dat_type_out = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.datatype;
    uint32_t dat_type_in = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.datatype;
    uint32_t dat_type_out_size = (dat_type_out == RINT8) ? 1 : (dat_type_out == RBFLOAT) ? 2
                                                                                         : 4;
    uint32_t size_out = dat_h_out * dat_w_out * dat_c_out;
    if ((line_stride_out > 0) || (surf_stride_out > 0))
    {
        if (dat_type_out != RINT8)
            return TYPE_ERR;
        output_addr_temp = (uint64_t)malloc(size_out * dat_type_out_size);
    }
    else
    {
        output_addr_temp = pDst;
    }

    uint64_t pSrc = cpu_dev.task->address_list[cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.index];
    uint32_t dat_type_in_size = (dat_type_in == RINT8) ? 1 : (dat_type_in == RBFLOAT) ? 2
                                                                                      : 4;
    uint32_t dat_h_in = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.height;
    uint32_t dat_w_in = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.width;
    uint32_t dat_c_in = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.channel;
    int32_t line_stride_in = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.line_stride;
    int32_t surf_stride_in = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.surf_stride;
    uint32_t size_in = dat_h_in * dat_w_in * dat_c_in;

    if (size_in != size_out)
    {
        debug_info("\nInput shape or output shape must be wrong!!!\n");
        return SHAPE_ERR;
    }

    if ((line_stride_in > 0) || (surf_stride_in > 0))
    {
        input_addr_temp = (uint64_t)malloc(size_in * dat_type_in_size);
        if (dat_type_in != RINT8)
        {
            debug_info("%s Wrong datatype!\n", __func__);
            return TYPE_ERR;
        }
        err = unvectorize((void *)pSrc, (void *)input_addr_temp, &cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data);
    }
    else
    {
        input_addr_temp = pSrc;
    }
    for (h = 0; h < dat_h_in; h++)
    {
        for (w = 0; w < dat_w_in; w++)
        {
            for (c = 0; c < dat_c_in; c++)
            {
                uint64_t input_addr = input_addr_temp + (c + w * dat_c_in + h * dat_w_in * dat_c_in) * dat_type_in_size;
                uint64_t output_addr = output_addr_temp + (c + w * dat_c_out + h * dat_w_out * dat_c_out) * dat_type_out_size;
                if ((dat_type_in == RINT8) && (dat_type_out == RFLOAT))
                {
                    float_t x = *(volatile int8_t *)input_addr * cpu_parameters.cpu_operation.common_only_op.common.input_scale_factor[0];
                    float_t y = 1 / (1 + expf(-x));
                    *(volatile float_t *)output_addr = y;
                }
                else if ((dat_type_in == RINT8) && (dat_type_out == RINT8))
                {
                    float_t x = *(volatile int8_t *)input_addr * cpu_parameters.cpu_operation.common_only_op.common.input_scale_factor[0];
                    float_t y = 1 / (1 + expf(-x));
                    y /= cpu_parameters.cpu_operation.common_only_op.common.output_scale_factor[0];
                    int8_t dstint8p = (y >= 127.0) ? 127 : (y >= -128.0) ? (int8_t)y
                                                                         : -128;
                    *(volatile int8_t *)output_addr = dstint8p;
                }
                else if ((dat_type_in == RFLOAT) && (dat_type_out == RFLOAT))
                {
                    float_t x = *(volatile float_t *)input_addr;
                    float_t y = 1 / (1 + expf(-x));
                    *(volatile float_t *)output_addr = y;
                }
                else if ((dat_type_in == RFLOAT) && (dat_type_out == RINT8))
                {
                    float_t x = *(volatile float_t *)input_addr;
                    float_t y = 1 / (1 + expf(-x));
                    y /= cpu_parameters.cpu_operation.common_only_op.common.output_scale_factor[0];
                    int8_t dstint8p = (y >= 127.0) ? 127 : (y >= -128.0) ? (int8_t)y
                                                                         : -128;
                    *(volatile int8_t *)output_addr = dstint8p;
                }
            }
        }
    }
    if ((line_stride_out > 0) || (surf_stride_out > 0))
    {
        err = vectorize((void *)output_addr_temp, (void *)pDst, &cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data);
        free((void *)output_addr_temp);
    }
    if ((line_stride_in > 0) || (surf_stride_in > 0))
    {
        free((void *)input_addr_temp);
    }
    // debug_info("\nExit %s;\n", __func__);
    // sifive_pl2cache0_flush((uintptr_t)pDst);
    cpu_output_debug(cpu_parameters);
    return err;
}

int32_t executeReshape(struct cpu_param cpu_parameters)
{
    // debug_info("Enter %s\n", __func__);
    int32_t err = 0;
    int32_t h, w, c;

    uint64_t output_addr_temp, input_addr_temp;
    uint32_t dat_h_out = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.height;
    uint32_t dat_w_out = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.width;
    uint32_t dat_c_out = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.channel;
    int32_t line_stride_out = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.line_stride;
    int32_t surf_stride_out = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.surf_stride;
    uint32_t dat_type_out = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.datatype;
    uint32_t dat_type_in = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.datatype;
    uint32_t dat_type_out_size = (dat_type_out == RINT8) ? 1 : (dat_type_out == RBFLOAT) ? 2
                                                                                         : 4;
    uint32_t size_out = dat_h_out * dat_w_out * dat_c_out;
    cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.address = cpu_dev.task->address_list[cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.index];
    uint64_t pSrc = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.address;

    cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.address = cpu_dev.task->address_list[cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.index];
    uint64_t pDst = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.address;
    uint32_t dat_type_in_size = (dat_type_in == RINT8) ? 1 : (dat_type_in == RBFLOAT) ? 2
                                                                                      : 4;
    uint32_t dat_h_in = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.height;
    uint32_t dat_w_in = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.width;
    uint32_t dat_c_in = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.channel;
    int32_t line_stride_in = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.line_stride;
    int32_t surf_stride_in = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.surf_stride;
    uint32_t size_in = dat_h_in * dat_w_in * dat_c_in;

    if (size_in != size_out)
    {
        debug_info("\nInput shape or output shape must be wrong!!!\n");
        return RESHAPE_ERR;
    }

    if ((dat_type_in == dat_type_out) && (dat_type_in != RINT8))
    {
        /* debug_info("pDst is 0x%08x, pSrc is 0x%08x, size is %d.\n", pDst, pSrc, size_out * dat_type_out_size); */
        if (pDst != pSrc)
            memcpy((void *)pDst, (void *)pSrc, size_out * dat_type_out_size);
        return err;
    }

    if ((line_stride_in > 0) || (surf_stride_in > 0))
    {
        input_addr_temp = (uint64_t)malloc(size_in * dat_type_in_size);
        if (dat_type_in != RINT8)
        {
            debug_info("%s Wrong datatype!\n", __func__);
            return TYPE_ERR;
        }
        err = unvectorize((void *)pSrc, (void *)input_addr_temp, &cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data);
    }
    else
    {
        input_addr_temp = pSrc;
    }

    if ((line_stride_out > 0) || (surf_stride_out > 0))
    {
        if (dat_type_out != RINT8)
            return TYPE_ERR;
        output_addr_temp = (uint64_t)malloc(size_out * dat_type_out_size);
    }
    else
    {
        output_addr_temp = pDst;
    }

    for (h = 0; h < dat_h_in; h++)
    {
        for (w = 0; w < dat_w_in; w++)
        {
            for (c = 0; c < dat_c_in; c++)
            {
                uint64_t input_addr = input_addr_temp + (c + w * dat_c_in + h * dat_w_in * dat_c_in) * dat_type_in_size;
                /* uint64_t output_addr = output_addr_temp + (c + w * dat_c_out + h * dat_w_out * dat_c_out) * dat_type_out_size; */
                uint64_t output_addr = output_addr_temp + (c + w * dat_c_in + h * dat_w_in * dat_c_in) * dat_type_out_size;
                if ((dat_type_in == RINT8) && (dat_type_out == RFLOAT))
                {
                    float_t x = *(volatile int8_t *)input_addr * cpu_parameters.cpu_operation.common_only_op.common.input_scale_factor[0];
                    float_t y = (x);
                    *(volatile float_t *)output_addr = y;
                }
                else if ((dat_type_in == RFLOAT) && (dat_type_out == RINT8))
                {
                    float_t x = *(volatile float_t *)input_addr;
                    float_t y = (x);
                    y /= cpu_parameters.cpu_operation.common_only_op.common.output_scale_factor[0];
                    int8_t dstint8p = (y >= 127.0) ? 127 : (y >= -128.0) ? (int8_t)y
                                                                         : -128;
                    *(volatile int8_t *)output_addr = dstint8p;
                }
                else if ((dat_type_in == RINT8) && (dat_type_out == RINT8))
                {
                    *(volatile int8_t *)output_addr = *(volatile int8_t *)input_addr;
                }
            }
        }
    }
    if ((line_stride_out > 0) || (surf_stride_out > 0))
    {
        err = vectorize((void *)output_addr_temp, (void *)pDst, &cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data);
        free((void *)output_addr_temp);
    }
    if ((line_stride_in > 0) || (surf_stride_in > 0))
    {
        free((void *)input_addr_temp);
    }
    cpu_output_debug(cpu_parameters);
    // debug_info("\nExit %s;\n", __func__);
    // sifive_pl2cache0_flush((uintptr_t)pDst);
    return err;
}

/*Input data shape must be vectorized*/
int32_t executeUpsampe_nearest(struct cpu_param cpu_parameters)
{
    uint32_t err = 0;
    uint64_t output_addr_temp, input_addr_temp;
    uint32_t dat_h_out = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.height;
    uint32_t dat_w_out = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.width;
    uint32_t dat_c_out = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.channel;
    int32_t line_stride_out = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.line_stride;
    int32_t surf_stride_out = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.surf_stride;
    uint32_t dat_type_out = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.datatype;
    uint32_t dat_type_in = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.datatype;
    uint32_t dat_type_out_size = (dat_type_out == RINT8) ? 1 : (dat_type_out == RBFLOAT) ? 2
                                                                                         : 4;
    uint32_t size_out = dat_h_out * dat_w_out * dat_c_out;

    cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.address = cpu_dev.task->address_list[cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.index];
    cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.address = cpu_dev.task->address_list[cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.index];

    uint64_t pSrc = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.address;
    uint64_t pDst = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.address;
    uint32_t dat_type_in_size = (dat_type_in == RINT8) ? 1 : (dat_type_in == RBFLOAT) ? 2
                                                                                      : 4;
    uint32_t dat_h_in = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.height;
    uint32_t dat_w_in = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.width;
    uint32_t dat_c_in = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.channel;
    int32_t line_stride_in = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.line_stride;
    int32_t surf_stride_in = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.surf_stride;
    uint32_t size_in = dat_h_in * dat_w_in * dat_c_in;
    uint32_t scale = dat_h_out / dat_h_in;
    /*
        if (dat_type_in != dat_type_out)
        {
            debug_info("Don't support CPU upsample operation with different src(%d) and dst(%d) formats\n", dat_type_in, dat_type_out);
            err = TYPE_ERR;
            return err;
        }
    */
    uint32_t h, w, c, sx, sy, surf;
    if (((line_stride_in > 0) || (surf_stride_in > 0)) &&
        ((line_stride_out > 0) || (surf_stride_out > 0)))
    {
        if ((dat_type_in != RINT8) || (dat_type_out != RINT8))
        {
            debug_info("\n%s datatype is Wrong\n", __func__);
            return TYPE_ERR;
        }
        for (surf = 0; surf < ((dat_c_in - 1) / C_ATM + 1); surf++)
        {
            for (h = 0; h < dat_h_in; h++)
            {
                for (w = 0; w < dat_w_in; w++)
                {
                    for (c = 0; c < C_ATM; c++)
                    {
                        uint64_t input_addr = pSrc + (c + w * C_ATM + h * line_stride_in + surf * surf_stride_in) * sizeof(int8_t);
                        int8_t input_data = *(volatile int8_t *)input_addr;
                        uint64_t output_addr = pDst + (c + w * scale * C_ATM + h * scale * line_stride_out + surf * surf_stride_out) * sizeof(int8_t);
                        for (sy = 0; sy < scale; sy++)
                        {
                            for (sx = 0; sx < scale; sx++)
                            {
                                int8_t *output_addr_temp = (int8_t *)(output_addr + (sx * C_ATM + sy * line_stride_out) * sizeof(int8_t));
                                *(volatile int8_t *)output_addr_temp = input_data;
                            }
                        }
                    }
                }
            }
        }
    }
    else
    {
        // first, change the vectorized data to unvectorize
        if ((line_stride_in > 0) || (surf_stride_in > 0))
        {
            input_addr_temp = (uint64_t)malloc(size_in * dat_type_in_size);
            err = unvectorize((void *)pSrc, (void *)input_addr_temp, &cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data);
        }
        else
        {
            input_addr_temp = pSrc;
        }
        // add data process code here
        if ((line_stride_out > 0) || (surf_stride_out > 0))
        {
            if (dat_type_out != RINT8)
                return TYPE_ERR;
            output_addr_temp = (uint64_t)malloc(size_out * dat_type_out_size);
        }
        else
        {
            output_addr_temp = pDst;
        }
        // debug_info("\nUpsample inputdata after unvectorize\n");
        for (h = 0; h < dat_h_in; h++)
        {
            // debug_info("\nh = %d\n", h);
            for (w = 0; w < dat_w_in; w++)
            {
                for (c = 0; c < dat_c_in; c++)
                {
                    uint64_t input_addr = input_addr_temp + (c + w * dat_c_in + h * dat_w_in * dat_c_in) * dat_type_in_size;
                    uint64_t output_addr = output_addr_temp + (c + scale * w * dat_c_in + scale * h * scale * dat_w_in * dat_c_in) * dat_type_out_size;
                    if (dat_type_out == RINT8)
                    {
                        int8_t input_data = *(volatile int8_t *)input_addr;
                        for (sy = 0; sy < scale; sy++)
                        {
                            for (sx = 0; sx < scale; sx++)
                            {
                                int8_t *output_addr_temp = (int8_t *)(output_addr + (sx * dat_c_in + sy * scale * dat_w_in * dat_c_in) * sizeof(int8_t));
                                *(volatile int8_t *)output_addr_temp = input_data;
                            }
                        }
                    }
                    else if (dat_type_out == RFLOAT)
                    {
                        float_t input_data = *(volatile float_t *)input_addr;
                        for (sy = 0; sy < scale; sy++)
                        {
                            for (sx = 0; sx < scale; sx++)
                            {
                                float_t *output_addr_temp = (float_t *)(output_addr + (sx * dat_c_in + sy * scale * dat_w_in * dat_c_in) * sizeof(float_t));
                                *(volatile float *)output_addr_temp = input_data;
                            }
                        }
                    }
                    else if (dat_type_out == RINT)
                    {
                        int32_t input_data = *(volatile int32_t *)input_addr;
                        for (sy = 0; sy < scale; sy++)
                        {
                            for (sx = 0; sx < scale; sx++)
                            {
                                int32_t *output_addr_temp = (int32_t *)(output_addr + (sx * dat_c_in + sy * scale * dat_w_in * dat_c_in) * sizeof(int32_t));
                                *(volatile int32_t *)output_addr_temp = input_data;
                            }
                        }
                    }
                    else
                    {
                        debug_info("%s unsupport datatype!\n", __func__);
                        err = TYPE_ERR;
                        return err;
                    }
                }
            }
        }
        // if the consumer is DLA, vectorize the output
        if ((line_stride_out > 0) || (surf_stride_out > 0))
        {
            err = vectorize((void *)output_addr_temp, (void *)pDst, &cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data);
            free((void *)output_addr_temp);
        }
        if ((line_stride_in > 0) || (surf_stride_in > 0))
        {
            free((void *)input_addr_temp);
        }
    }
    // debug_info("\nExit %s;\n", __func__);
    // sifive_pl2cache0_flush((uintptr_t)pDst);
    cpu_output_debug(cpu_parameters);
    return err;
}

int32_t executeConcat(struct cpu_param cpu_parameters)
{
    uint32_t i, h, w, c, err = 0;
    uint64_t input_addr_temp, output_addr_temp;
    uint32_t input_num = cpu_parameters.cpu_operation.concat_op.common.input_num;
    if (input_num > INPUT_MAX)
    {
        return INPUT_ERR;
    }
    uint32_t h_offset = 0;
    uint32_t w_offset = 0;
    uint32_t c_offset = 0;
    uint64_t pDst = (uint64_t)cpu_dev.task->address_list[cpu_parameters.cpu_operation_buffer.concat_buffers.dst_data.index];
    /* uint64_t pDst = cpu_parameters.cpu_operation_buffer.concat_buffers.dst_data.address; */
    uint32_t dat_h_out = cpu_parameters.cpu_operation_buffer.concat_buffers.dst_data.height;
    uint32_t dat_w_out = cpu_parameters.cpu_operation_buffer.concat_buffers.dst_data.width;
    uint32_t dat_c_out = cpu_parameters.cpu_operation_buffer.concat_buffers.dst_data.channel;
    int32_t line_stride_out = cpu_parameters.cpu_operation_buffer.concat_buffers.dst_data.line_stride;
    int32_t surf_stride_out = cpu_parameters.cpu_operation_buffer.concat_buffers.dst_data.surf_stride;
    uint32_t dat_type_out = cpu_parameters.cpu_operation_buffer.concat_buffers.dst_data.datatype;
    uint32_t dat_type_out_size = (dat_type_out == RINT8) ? 1 : (dat_type_out == RBFLOAT) ? 2
                                                                                         : 4;
    uint32_t size_out = dat_h_out * dat_w_out * dat_c_out;
    uint32_t axis = cpu_parameters.cpu_operation.concat_op.axis;
    // debug_info("Output address is 0x%08x.\n", pDst);
    // debug_info("\tOutput shape is  %d x %d x %d, stride is %d x %d.\n", dat_h_out, dat_w_out, dat_c_out, line_stride_out, surf_stride_out);
    if ((line_stride_out > 0) || (surf_stride_out > 0))
    {
        if (dat_type_out != RINT8)
            return TYPE_ERR;
        output_addr_temp = (uint64_t)malloc(size_out * dat_type_out_size);
    }
    else
    {
        output_addr_temp = pDst;
    }

    for (i = 0; i < input_num; i++)
    {
        uint64_t pSrc = (uint64_t)cpu_dev.task->address_list[cpu_parameters.cpu_operation.concat_op.src_data[i].index];
        /* uint64_t pSrc = cpu_parameters.cpu_operation.concat_op.src_data[i].address; */
        // debug_info("\t %d input address is 0x%08x.\n", i, pSrc); int32_t j = 0;
        uint32_t dat_h_in = cpu_parameters.cpu_operation.concat_op.src_data[i].height;
        uint32_t dat_w_in = cpu_parameters.cpu_operation.concat_op.src_data[i].width;
        uint32_t dat_c_in = cpu_parameters.cpu_operation.concat_op.src_data[i].channel;
        int32_t line_stride_in = cpu_parameters.cpu_operation.concat_op.src_data[i].line_stride;
        int32_t surf_stride_in = cpu_parameters.cpu_operation.concat_op.src_data[i].surf_stride;
        uint32_t dat_type_in = cpu_parameters.cpu_operation.concat_op.src_data[i].datatype;
        uint32_t dat_type_in_size = (dat_type_in == RINT8) ? 1 : (dat_type_in == RBFLOAT) ? 2
                                                                                          : 4;
        uint32_t size_in = dat_h_in * dat_w_in * dat_c_in;

        /* debug_info("src_data: \n" */
        /*            "\t data_type:%u;\n" */
        /*            "\t height : %u\n" */
        /*            "\t width : %u\n" */
        /*            "\t channel: %u\n" */
        /*            "\t line_stride : %d\n" */
        /*            "\t surf_stride: %d\n" */
        /*            "\t size: %u\n" */
        /*            "\t index: %u\n" */
        /*            "\t address: 0x%08x\n", */
        /*            dat_type_in, */
        /*            dat_h_in, */
        /*            dat_w_in, */
        /*            dat_c_in, */
        /*            line_stride_in, */
        /*            surf_stride_in, */
        /*            cpu_parameters.cpu_operation.concat_op.src_data[i].size, */
        /*            cpu_parameters.cpu_operation.concat_op.src_data[i].index, */
        /*            pSrc); */

        // debug_info("\t%d input shape is %d x %d x %d, stride is %d x %d.\n", i, dat_h_in, dat_w_in, dat_c_in, line_stride_in, surf_stride_in);
        /*unvectorize the data from dla */
        if ((line_stride_in > 0) || (surf_stride_in > 0))
        {
            // debug_info("\t Dla input.\n");
            if (dat_type_in != RINT8)
                return TYPE_ERR;
            input_addr_temp = (uint64_t)malloc(size_in * sizeof(int8_t));
            err = unvectorize((void *)(pSrc), (void *)input_addr_temp, &cpu_parameters.cpu_operation.concat_op.src_data[i]);
        }
        else
        {
            input_addr_temp = pSrc;
        }
        float_t iscale = cpu_parameters.cpu_operation.concat_op.common.input_scale_factor[i];
        float_t oscale = cpu_parameters.cpu_operation.concat_op.common.output_scale_factor[0];
        // debug_info("\tAxis %d, Input data type %d, output data type %d, Input scale factor and output scale factor is:\n", axis, dat_type_in, dat_type_out);
        // debug_info("\t"); print_float(iscale); print_float(oscale);
        // debug_info("\n %d \n", i);
        /*if the input data type is int8, the inverse quantization will be needed */
        for (h = 0; h < dat_h_in; h++)
        {
            for (w = 0; w < dat_w_in; w++)
            {
                for (c = 0; c < dat_c_in; c++)
                {
                    uint64_t input_addr = input_addr_temp + (c + w * dat_c_in + h * dat_w_in * dat_c_in) * dat_type_in_size;
                    if (axis == 3)
                    {
                        uint64_t output_addr = output_addr_temp + (c_offset + c + w * dat_c_out + h * dat_w_out * dat_c_out) * dat_type_out_size;
                        if ((dat_type_in == RINT8) && (dat_type_out == RFLOAT))
                        {
                            float_t input_data = *(volatile int8_t *)input_addr * iscale;
                            *(volatile float_t *)output_addr = input_data;
                        }
                        else if ((dat_type_in == RINT8) && (dat_type_out == RINT8))
                        {
                            float_t input_data = *(volatile int8_t *)input_addr * iscale / oscale;
                            int8_t dstint8p = (input_data >= 127.0) ? 127 : (input_data >= -128.0) ? (int8_t)input_data
                                                                                                   : -128;
                            *(volatile int8_t *)output_addr = dstint8p;
                        }
                        else if (dat_type_in == RINT)
                        {
                            int32_t input_data = *(volatile int32_t *)input_addr;
                            *(volatile int32_t *)output_addr = input_data;
                        }
                        else if ((dat_type_in == RFLOAT) && (dat_type_out == RFLOAT))
                        {
                            float_t input_data = *(volatile float_t *)input_addr;
                            *(volatile float_t *)output_addr = input_data;
                        }
                        else if ((dat_type_in == RFLOAT) && (dat_type_out == RINT8))
                        {
                            float_t input_data = *(volatile float_t *)input_addr / oscale;
                            int8_t dstint8p = (input_data >= 127.0) ? 127 : (input_data >= -128.0) ? (int8_t)input_data
                                                                                                   : -128;
                            *(volatile int8_t *)output_addr = dstint8p;
                        }
                        /*
                        if ((output_addr_temp == 0x84647020) && (j++ < 32))
                            print_float(*(volatile float *)input_addr);
                        if ((output_addr_temp == 0x853a9520) && (j++ < 32))
                            print_float(*(volatile float *)input_addr);
                        if ((output_addr_temp == 0x85702100) && (j++ < 32))
                            print_float(*(volatile float *)input_addr);i
                        */
                    }
                    else if (axis == 2)
                    {
                        uint64_t output_addr = output_addr_temp + (c + (w + w_offset) * dat_c_out + h * dat_w_out * dat_c_out) * dat_type_out_size;
                        if ((dat_type_in == RINT8) && (dat_type_out == RFLOAT))
                        {
                            float_t input_data = *(volatile int8_t *)input_addr * iscale;
                            *(volatile float_t *)output_addr = input_data;
                        }
                        else if ((dat_type_in == RINT8) && (dat_type_out == RINT8))
                        {
                            float_t input_data = *(volatile int8_t *)input_addr * iscale / oscale;
                            int8_t dstint8p = (input_data >= 127.0) ? 127 : (input_data >= -128.0) ? (int8_t)input_data
                                                                                                   : -128;
                            *(volatile int8_t *)output_addr = dstint8p;
                        }
                        else if (dat_type_in == RINT)
                        {
                            int32_t input_data = *(volatile int32_t *)input_addr;
                            *(volatile int32_t *)output_addr = input_data;
                        }
                        else if ((dat_type_in == RFLOAT) && (dat_type_out == RFLOAT))
                        {
                            float_t input_data = *(volatile float_t *)input_addr;
                            *(volatile float_t *)output_addr = input_data;
                        }
                        else if ((dat_type_in == RFLOAT) && (dat_type_out == RINT8))
                        {
                            float_t input_data = *(volatile float_t *)input_addr / oscale;
                            int8_t dstint8p = (input_data >= 127.0) ? 127 : (input_data >= -128.0) ? (int8_t)input_data
                                                                                                   : -128;
                            *(volatile int8_t *)output_addr = dstint8p;
                        }
                    }
                    else if (axis == 1)
                    {
                        uint64_t output_addr = output_addr_temp + (c + w * dat_c_out + (h + h_offset) * dat_w_out * dat_c_out) * dat_type_out_size;
                        if ((dat_type_in == RINT8) && (dat_type_out == RFLOAT))
                        {
                            float_t input_data = *(volatile int8_t *)input_addr * iscale;
                            *(volatile float_t *)output_addr = input_data;
                        }
                        else if ((dat_type_in == RINT8) && (dat_type_out == RINT8))
                        {
                            float_t input_data = *(volatile int8_t *)input_addr * iscale / oscale;
                            int8_t dstint8p = (input_data >= 127.0) ? 127 : (input_data >= -128.0) ? (int8_t)input_data
                                                                                                   : -128;
                            *(volatile int8_t *)output_addr = dstint8p;
                        }
                        else if (dat_type_in == RINT)
                        {
                            int32_t input_data = *(volatile int32_t *)input_addr;
                            *(volatile int32_t *)output_addr = input_data;
                        }
                        else if ((dat_type_in == RFLOAT) && (dat_type_out == RFLOAT))
                        {
                            float_t input_data = *(volatile float_t *)input_addr;
                            *(volatile float_t *)output_addr = input_data;
                        }
                        else if ((dat_type_in == RFLOAT) && (dat_type_out == RINT8))
                        {
                            float_t input_data = *(volatile float_t *)input_addr / oscale;
                            int8_t dstint8p = (input_data >= 127.0) ? 127 : (input_data >= -128.0) ? (int8_t)input_data
                                                                                                   : -128;
                            *(volatile int8_t *)output_addr = dstint8p;
                        }
                    }
                    else
                    {
                        debug_info("%s unsupported axis %d !\n", __func__, axis);
                        err = CONCAT_ERR;
                    }
                }
            }
        }
        h_offset += dat_h_in;
        w_offset += dat_w_in;
        c_offset += dat_c_in;
        if ((line_stride_in > 0) || (surf_stride_in > 0))
            free((void *)input_addr_temp);
    }
    if ((line_stride_out > 0) || (surf_stride_out > 0))
    {
        vectorize((void *)output_addr_temp, (void *)pDst, &cpu_parameters.cpu_operation_buffer.concat_buffers.dst_data);
        free((void *)output_addr_temp);
    }
    cpu_output_debug(cpu_parameters);
    // debug_info("\nExit %s;\n", __func__);
    // sifive_pl2cache0_flush((uintptr_t)pDst);
    return err;
}

int32_t executeSlice(struct cpu_param cpu_parameters)
{
    uint32_t h, w, c, err = 0;
    uint64_t input_addr_temp, output_addr_temp;
    uint64_t pDst = cpu_dev.task->address_list[cpu_parameters.cpu_operation_buffer.slice_buffers.dst_data.index];
    uint32_t dat_h_out = cpu_parameters.cpu_operation_buffer.slice_buffers.dst_data.height;
    uint32_t dat_w_out = cpu_parameters.cpu_operation_buffer.slice_buffers.dst_data.width;
    uint32_t dat_c_out = cpu_parameters.cpu_operation_buffer.slice_buffers.dst_data.channel;
    int32_t line_stride_out = cpu_parameters.cpu_operation_buffer.slice_buffers.dst_data.line_stride;
    int32_t surf_stride_out = cpu_parameters.cpu_operation_buffer.slice_buffers.dst_data.surf_stride;
    uint32_t dat_type_out = cpu_parameters.cpu_operation_buffer.slice_buffers.dst_data.datatype;
    uint32_t dat_type_in = cpu_parameters.cpu_operation_buffer.slice_buffers.src_data.datatype;
    uint32_t dat_type_out_size = (dat_type_out == RINT8) ? 1 : (dat_type_out == RBFLOAT) ? 2
                                                                                         : 4;
    uint32_t size_out = dat_h_out * dat_w_out * dat_c_out;
    if ((line_stride_out > 0) || (surf_stride_out > 0))
    {
        if (dat_type_out != RINT8)
            return TYPE_ERR;
        output_addr_temp = (uint64_t)malloc(size_out * dat_type_out_size);
    }
    else
    {
        output_addr_temp = pDst;
    }

    uint64_t pSrc = cpu_dev.task->address_list[cpu_parameters.cpu_operation_buffer.slice_buffers.src_data.index];
    uint32_t dat_type_in_size = (dat_type_in == RINT8) ? 1 : (dat_type_in == RBFLOAT) ? 2
                                                                                      : 4;
    uint32_t dat_h_in = cpu_parameters.cpu_operation_buffer.slice_buffers.src_data.height;
    uint32_t dat_w_in = cpu_parameters.cpu_operation_buffer.slice_buffers.src_data.width;
    uint32_t dat_c_in = cpu_parameters.cpu_operation_buffer.slice_buffers.src_data.channel;
    int32_t line_stride_in = cpu_parameters.cpu_operation_buffer.slice_buffers.src_data.line_stride;
    int32_t surf_stride_in = cpu_parameters.cpu_operation_buffer.slice_buffers.src_data.surf_stride;
    uint32_t size_in = dat_h_in * dat_w_in * dat_c_in;

    uint32_t h_start = cpu_parameters.cpu_operation.slice_op.begin[1];
    uint32_t w_start = cpu_parameters.cpu_operation.slice_op.begin[2];
    uint32_t c_start = cpu_parameters.cpu_operation.slice_op.begin[3];

    uint32_t h_end = cpu_parameters.cpu_operation.slice_op.end[1];
    uint32_t w_end = cpu_parameters.cpu_operation.slice_op.end[2];
    uint32_t c_end = cpu_parameters.cpu_operation.slice_op.end[3];

    uint32_t h_stride = cpu_parameters.cpu_operation.slice_op.stride[1];
    uint32_t w_stride = cpu_parameters.cpu_operation.slice_op.stride[2];
    uint32_t c_stride = cpu_parameters.cpu_operation.slice_op.stride[3];
    //debug_info("\n\tStart %ux%ux%u, End %ux%ux%u, Stride %ux%ux%u;\n", h_start, w_start, c_start,\
    h_end, w_end, c_end, h_stride, w_stride, c_stride);
    if ((line_stride_in > 0) || (surf_stride_in > 0))
    {
        input_addr_temp = (uint64_t)malloc(size_in * dat_type_in_size);
        if (dat_type_in != RINT8)
        {
            debug_info("%s Wrong datatype!\n", __func__);
            return TYPE_ERR;
        }
        err = unvectorize((void *)pSrc, (void *)input_addr_temp, &cpu_parameters.cpu_operation_buffer.slice_buffers.src_data);
    }
    else
    {
        input_addr_temp = pSrc;
    }

    uint64_t output_addr = output_addr_temp;
    for (h = h_start; h < h_end; h += h_stride)
    {
        for (w = w_start; w < w_end; w += w_stride)
        {
            for (c = c_start; c < c_end; c += c_stride)
            {
                uint64_t input_addr = input_addr_temp + (c + w * dat_c_in + h * dat_w_in * dat_c_in) * dat_type_in_size;
                // if (dat_type_in == RINT8)
                if ((dat_type_in == RINT8) && (dat_type_out == RFLOAT))
                {
                    float_t x = *(volatile int8_t *)input_addr * cpu_parameters.cpu_operation.common_only_op.common.input_scale_factor[0];
                    float_t y = (x);
                    *(volatile float_t *)output_addr = y;
                    output_addr += 4;
                    // int8_t input_data = *(volatile int8_t *)input_addr;
                    // *(volatile int8_t *)output_addr = input_data;
                    // output_addr += 1;
                }
                else if ((dat_type_in == RFLOAT) && (dat_type_out == RFLOAT))
                {
                    float_t input_data = *(volatile float_t *)input_addr;
                    *(volatile float_t *)output_addr = input_data;
                    output_addr += 4;
                }
                else if (dat_type_in == RINT)
                {
                    int32_t input_data = *(volatile int32_t *)input_addr;
                    *(volatile int32_t *)output_addr = input_data;
                    output_addr += 4;
                }
                else if ((dat_type_in == RINT8) && (dat_type_out == RINT8))
                {
                    int8_t input_data = *(volatile int8_t *)input_addr;
                    *(volatile uint8_t *)output_addr = input_data;
                    output_addr += 1;
                }
                else if ((dat_type_in == RFLOAT) && (dat_type_out == RINT8))
                {
                    float_t input_data = *(volatile float_t *)input_addr;
                    float_t y = input_data / cpu_parameters.cpu_operation.split_op.common.output_scale_factor[0];
                    int8_t dstint8p = (y >= 127.0) ? 127 : (y >= -128.0) ? (int8_t)y
                                                                         : -128;
                    *(volatile int8_t *)output_addr = dstint8p;
                    output_addr += 1;
                }
                else
                {
                    debug_info("\n%s unsupport datatype %d\n", __func__, dat_type_in);
                }
            }
        }
    }

    if ((line_stride_out > 0) || (surf_stride_out > 0))
    {
        err = vectorize((void *)output_addr_temp, (void *)pDst, &cpu_parameters.cpu_operation_buffer.slice_buffers.dst_data);
        free((void *)output_addr_temp);
    }
    if ((line_stride_in > 0) || (surf_stride_in > 0))
        free((void *)input_addr_temp);
    cpu_output_debug(cpu_parameters);
    // debug_info("\nExit %s;\n", __func__);
    // sifive_pl2cache0_flush((uintptr_t)pDst);
    return err;
}

int32_t executeAddV2_Sub_Mult_Div_Power(struct cpu_param cpu_parameters)
{
    uint32_t h, w, c, dat_tmp, err = 0;
    uint64_t pDst = cpu_dev.task->address_list[cpu_parameters.cpu_operation_buffer.with_weight_op_buffers.dst_data.index];
    uint64_t output_addr_temp, input_addr_temp0, input_addr_temp1;
    uint32_t dat_h_out = cpu_parameters.cpu_operation_buffer.with_weight_op_buffers.dst_data.height;
    uint32_t dat_w_out = cpu_parameters.cpu_operation_buffer.with_weight_op_buffers.dst_data.width;
    uint32_t dat_c_out = cpu_parameters.cpu_operation_buffer.with_weight_op_buffers.dst_data.channel;
    int32_t line_stride_out = cpu_parameters.cpu_operation_buffer.with_weight_op_buffers.dst_data.line_stride;
    int32_t surf_stride_out = cpu_parameters.cpu_operation_buffer.with_weight_op_buffers.dst_data.surf_stride;
    uint32_t dat_type_out = cpu_parameters.cpu_operation_buffer.with_weight_op_buffers.dst_data.datatype;
    uint32_t dat_type_out_size = (dat_type_out == RINT8) ? 1 : (dat_type_out == RBFLOAT) ? 2
                                                                                         : 4;
    uint32_t size_out = dat_h_out * dat_w_out * dat_c_out;
    if ((line_stride_out > 0) || (surf_stride_out > 0))
    {
        if (dat_type_out != RINT8)
            return TYPE_ERR;
        output_addr_temp = (uint64_t)malloc(size_out * dat_type_out_size);
    }
    else
    {
        output_addr_temp = pDst;
    }
    uint64_t pSrc0 = cpu_dev.task->address_list[cpu_parameters.cpu_operation_buffer.with_weight_op_buffers.src_data.index];
    uint64_t pSrc1 = cpu_dev.task->address_list[cpu_parameters.cpu_operation.with_weight_op.weight.index];
    uint32_t dat_type = cpu_parameters.cpu_operation_buffer.with_weight_op_buffers.src_data.datatype;
    uint32_t weight_type = cpu_parameters.cpu_operation.with_weight_op.weight.datatype;
    uint32_t dat_type_in_size = (dat_type == RINT8) ? 1 : (dat_type == RBFLOAT) ? 2
                                                                                : 4;

    uint32_t dat_h = cpu_parameters.cpu_operation_buffer.with_weight_op_buffers.src_data.height;
    uint32_t dat_w = cpu_parameters.cpu_operation_buffer.with_weight_op_buffers.src_data.width;
    uint32_t dat_c = cpu_parameters.cpu_operation_buffer.with_weight_op_buffers.src_data.channel;
    int32_t dat_line_stride = cpu_parameters.cpu_operation_buffer.with_weight_op_buffers.src_data.line_stride;
    int32_t dat_surf_stride = cpu_parameters.cpu_operation_buffer.with_weight_op_buffers.src_data.surf_stride;
    uint32_t dat_size = dat_h * dat_w * dat_c;

    uint32_t wt_h = cpu_parameters.cpu_operation.with_weight_op.weight.height;
    uint32_t wt_w = cpu_parameters.cpu_operation.with_weight_op.weight.width;
    uint32_t wt_c = cpu_parameters.cpu_operation.with_weight_op.weight.channel;
    int32_t wt_line_stride = cpu_parameters.cpu_operation.with_weight_op.weight.line_stride;
    int32_t wt_surf_stride = cpu_parameters.cpu_operation.with_weight_op.weight.surf_stride;
    uint32_t wt_size = wt_h * wt_w * wt_c;

    /* /1* uint32_t add_type = cpu_parameters.cpu_operation.with_weight_op.sub_op_type; *1/ */
    /* if ((dat_h != wt_h) || (dat_w != wt_w) || (dat_c != wt_c)) { */
    /*     debug_info("\n%s: Warning: Data shape(%d x %d x %d) is different with weight(%d x %d x %d).\n", __func__, */
    /*                dat_h, dat_w, dat_c, wt_h, wt_w, wt_c); */
    /* } */

    if ((dat_line_stride > 0) || (dat_surf_stride > 0))
    {
        if (dat_type != RINT8)
        {
            debug_info("Error, %s Wrong datatype!\n", __func__);
            return TYPE_ERR;
        }
        void *temp0 = (void *)malloc(dat_size * dat_type_in_size);
        err = unvectorize((void *)pSrc0, (void *)temp0, &cpu_parameters.cpu_operation_buffer.with_weight_op_buffers.src_data);
        input_addr_temp0 = (uint64_t)malloc(dat_size * sizeof(float));
        for (h = 0; h < dat_h; h++)
        {
            for (w = 0; w < dat_w; w++)
            {
                for (c = 0; c < dat_c; c++)
                {
                    *(volatile float *)(input_addr_temp0 + (c + w * dat_c + h * dat_w * dat_c) * sizeof(float)) =
                        *(volatile int8_t *)((uint64_t)temp0 + (c + w * dat_c + h * dat_w * dat_c) * sizeof(int8_t)) *
                        cpu_parameters.cpu_operation.with_weight_op.common.input_scale_factor[0];
                }
            }
        }
        free(temp0);
    }
    else
    {
        input_addr_temp0 = pSrc0;
    }
    // debug_info("%s: dat_line_stride is %d, dat_surf_stride is %d, input_addr_temp0 is 0x%08x.\n", __func__, dat_line_stride, dat_surf_stride, input_addr_temp0);

    if ((wt_line_stride > 0) || (wt_surf_stride > 0))
    {
        if (weight_type != RINT8)
        {
            debug_info("Error, %s Wrong datatype!\n", __func__);
            return TYPE_ERR;
        }
        void *temp1 = (void *)malloc(wt_size * dat_type_in_size);
        err = unvectorize((void *)pSrc1, (void *)temp1, &cpu_parameters.cpu_operation.with_weight_op.weight);
        input_addr_temp1 = (uint64_t)malloc(wt_size * sizeof(float));
        for (h = 0; h < wt_h; h++)
        {
            for (w = 0; w < wt_w; w++)
            {
                for (c = 0; c < wt_c; c++)
                {
                    *(volatile float *)(input_addr_temp1 + (c + w * wt_c + h * wt_w * wt_c) * sizeof(float)) =
                        *(volatile int8_t *)((uint64_t)temp1 + (c + w * wt_c + h * wt_w * wt_c) * sizeof(int8_t)) *
                        cpu_parameters.cpu_operation.with_weight_op.common.input_scale_factor[1];
                }
            }
        }
        free(temp1);
    }
    else
    {
        input_addr_temp1 = pSrc1;
    }

    uint64_t output_addr = output_addr_temp;
    uint32_t wt_expend_type = (wt_size == 1) ? BROADCAST_EXPEND : (wt_size == wt_c)      ? C_EXPEND
                                                              : (wt_size == wt_w)        ? W_EXPEND
                                                              : (wt_size == wt_h)        ? H_EXPEND
                                                              : (wt_size == wt_h * wt_c) ? H_C_EXPEND
                                                              : (wt_size == wt_w * wt_c) ? W_C_EXPEND
                                                              : (wt_size == wt_h * wt_w) ? H_W_EXPEND
                                                                                         : NO_EXPEND;
    uint32_t dat_expend_type = (dat_size == 1) ? BROADCAST_EXPEND : (dat_size == dat_c)       ? C_EXPEND
                                                                : (dat_size == dat_w)         ? W_EXPEND
                                                                : (dat_size == dat_h)         ? H_EXPEND
                                                                : (dat_size == dat_h * dat_c) ? H_C_EXPEND
                                                                : (dat_size == dat_w * dat_c) ? W_C_EXPEND
                                                                : (dat_size == dat_h * dat_w) ? H_W_EXPEND
                                                                                              : NO_EXPEND;
    // debug_info(" output_addr 0x%08x\n", output_addr);
    for (h = 0; h < dat_h_out; h++)
    {
        for (w = 0; w < dat_w_out; w++)
        {
            for (c = 0; c < dat_c_out; c++)
            {
                void *input_addr0 = (void *)((dat_expend_type == NO_EXPEND) ? (input_addr_temp0 + (c + w * dat_c + h * dat_w * dat_c) * sizeof(float)) : (dat_expend_type == W_EXPEND) ? (input_addr_temp0 + w * sizeof(float))
                                                                                                                                                     : (dat_expend_type == H_EXPEND)   ? (input_addr_temp0 + h * sizeof(float))
                                                                                                                                                     : (dat_expend_type == C_EXPEND)   ? (input_addr_temp0 + c * sizeof(float))
                                                                                                                                                     : (dat_expend_type == H_C_EXPEND) ? (input_addr_temp0 + (c + h * dat_c) * sizeof(float))
                                                                                                                                                     : (dat_expend_type == W_C_EXPEND) ? (input_addr_temp0 + (c + w * dat_c) * sizeof(float))
                                                                                                                                                     : (dat_expend_type == H_W_EXPEND) ? (input_addr_temp0 + (w + h * dat_w) * sizeof(float))
                                                                                                                                                                                       : input_addr_temp0);
                void *input_addr1 = (void *)((wt_expend_type == NO_EXPEND) ? (input_addr_temp1 + (c + w * wt_c + h * wt_w * wt_c) * sizeof(float)) : (wt_expend_type == W_EXPEND) ? (input_addr_temp1 + w * sizeof(float))
                                                                                                                                                 : (wt_expend_type == H_EXPEND)   ? (input_addr_temp1 + h * sizeof(float))
                                                                                                                                                 : (wt_expend_type == C_EXPEND)   ? (input_addr_temp1 + c * sizeof(float))
                                                                                                                                                 : (wt_expend_type == H_C_EXPEND) ? (input_addr_temp1 + (c + h * wt_c) * sizeof(float))
                                                                                                                                                 : (wt_expend_type == W_C_EXPEND) ? (input_addr_temp1 + (c + w * wt_c) * sizeof(float))
                                                                                                                                                 : (wt_expend_type == H_W_EXPEND) ? (input_addr_temp1 + (w + h * wt_w) * sizeof(float))
                                                                                                                                                                                  : input_addr_temp1);
                // debug_info("Dat type is %d.\n", dat_type);
                if (dat_type_out == RINT8)
                {
                    float y = 0.0;
                    float input_data0 = *(volatile float *)input_addr0;
                    float input_data1 = *(volatile float *)input_addr1;
                    if (cpu_parameters.cpu_operation.with_weight_op.common.op_type == ADD)
                        y = (input_data0 + input_data1) / cpu_parameters.cpu_operation.with_weight_op.common.output_scale_factor[0];
                    else if (cpu_parameters.cpu_operation.with_weight_op.common.op_type == SUBTRACT)
                        y = (input_data0 - input_data1) / cpu_parameters.cpu_operation.with_weight_op.common.output_scale_factor[0];
                    else if (cpu_parameters.cpu_operation.with_weight_op.common.op_type == MULTIPLY)
                        y = (input_data0 * input_data1) / cpu_parameters.cpu_operation.with_weight_op.common.output_scale_factor[0];
                    else if (cpu_parameters.cpu_operation.with_weight_op.common.op_type == DIVIDE)
                        y = (input_data0 / input_data1) / cpu_parameters.cpu_operation.with_weight_op.common.output_scale_factor[0];
                    else if (cpu_parameters.cpu_operation.with_weight_op.common.op_type == POWER)
                        y = powf(input_data0, input_data1) / cpu_parameters.cpu_operation.with_weight_op.common.output_scale_factor[0];
                    else
                        debug_info("Error, in %s : %d, op_type error!\n", __func__, __LINE__);
                    int8_t dstint8p = (y >= 127.0) ? 127 : (y >= -128.0) ? (int8_t)y
                                                                         : -128;
                    *(volatile int8_t *)output_addr = dstint8p;
                    output_addr += sizeof(int8_t);
                }
                else if (dat_type_out == RFLOAT)
                {
                    float output_data = 0.0;
                    // debug_info("%d: address0 - 0x%08x, address1 - 0x%08x: ", (c + w * dat_c + h * dat_w * dat_c), input_addr0, input_addr1);
                    float input_data0 = *(volatile float_t *)input_addr0;
                    float input_data1 = *(volatile float_t *)input_addr1;
                    if (cpu_parameters.cpu_operation.with_weight_op.common.op_type == ADD)
                        output_data = input_data0 + input_data1;
                    else if (cpu_parameters.cpu_operation.with_weight_op.common.op_type == SUBTRACT)
                        output_data = input_data0 - input_data1;
                    else if (cpu_parameters.cpu_operation.with_weight_op.common.op_type == MULTIPLY)
                        output_data = input_data0 * input_data1;
                    else if (cpu_parameters.cpu_operation.with_weight_op.common.op_type == DIVIDE)
                        output_data = input_data0 / input_data1;
                    else if (cpu_parameters.cpu_operation.with_weight_op.common.op_type == POWER)
                        output_data = powf(input_data0, input_data1);
                    else
                        debug_info("Error, in %s : %d, op_type error!\n", __func__, __LINE__);
                    // print_float(input_data0);
                    // debug_info(" * ");
                    // print_float(input_data1);
                    // debug_info(" output_addr 0x%08x\n", output_addr);
                    *(volatile float *)output_addr = output_data;
                    output_addr += sizeof(float);
                }
                else
                {
                    debug_info("\nWarning: %s Unsupport data type\n", __func__);
                }
            }
        }
    }

    if ((line_stride_out > 0) || (surf_stride_out > 0))
    {
        err = vectorize((void *)output_addr_temp, (void *)pDst, &cpu_parameters.cpu_operation_buffer.with_weight_op_buffers.dst_data);
        free((void *)output_addr_temp);
    }
    if ((dat_line_stride > 0) || (dat_surf_stride > 0))
        free((void *)input_addr_temp0);
    if ((wt_line_stride > 0) || (wt_surf_stride > 0))
        free((void *)input_addr_temp1);
    // sifive_pl2cache0_flush((uintptr_t)pDst);
    // debug_info("\nExit %s;\n", __func__);
    cpu_output_debug(cpu_parameters);
    return err;
}

int32_t executeCast(struct cpu_param cpu_parameters)
{
    // debug_info("Enter %s\n", __func__);
    int32_t err = 0;
    int32_t h, w, c;

    uint64_t output_addr_temp, input_addr_temp;
    uint64_t pDst = cpu_dev.task->address_list[cpu_parameters.cpu_operation_buffer.cast_buffers.dst_data.index];
    uint32_t dat_h_out = cpu_parameters.cpu_operation_buffer.cast_buffers.dst_data.height;
    uint32_t dat_w_out = cpu_parameters.cpu_operation_buffer.cast_buffers.dst_data.width;
    uint32_t dat_c_out = cpu_parameters.cpu_operation_buffer.cast_buffers.dst_data.channel;
    int32_t line_stride_out = cpu_parameters.cpu_operation_buffer.cast_buffers.dst_data.line_stride;
    int32_t surf_stride_out = cpu_parameters.cpu_operation_buffer.cast_buffers.dst_data.surf_stride;
    uint32_t dat_type_out = cpu_parameters.cpu_operation_buffer.cast_buffers.dst_data.datatype;
    uint32_t dat_type_in = cpu_parameters.cpu_operation_buffer.cast_buffers.src_data.datatype;
    uint32_t dat_type_out_size = (dat_type_out == RINT8) ? 1 : (dat_type_out == RBFLOAT) ? 2
                                                                                         : 4;
    uint32_t size_out = dat_h_out * dat_w_out * dat_c_out;

    uint64_t pSrc = cpu_dev.task->address_list[cpu_parameters.cpu_operation_buffer.cast_buffers.src_data.index];
    uint32_t dat_type_in_size = (dat_type_in == RINT8) ? 1 : (dat_type_in == RBFLOAT) ? 2
                                                                                      : 4;
    uint32_t dat_h_in = cpu_parameters.cpu_operation_buffer.cast_buffers.src_data.height;
    uint32_t dat_w_in = cpu_parameters.cpu_operation_buffer.cast_buffers.src_data.width;
    uint32_t dat_c_in = cpu_parameters.cpu_operation_buffer.cast_buffers.src_data.channel;
    int32_t line_stride_in = cpu_parameters.cpu_operation_buffer.cast_buffers.src_data.line_stride;
    int32_t surf_stride_in = cpu_parameters.cpu_operation_buffer.cast_buffers.src_data.surf_stride;
    uint32_t size_in = dat_h_in * dat_w_in * dat_c_in;

    uint32_t dat_type = cpu_parameters.cpu_operation.cast_op.datatype;

    if (size_in != size_out)
    {
        debug_info("\nError, input shape or output shape must be wrong!!!\n");
        return RESHAPE_ERR;
    }

    if (dat_type_in == dat_type_out)
    {
        /* debug_info("Warning : datatype are same! pDst is 0x%08x, pSrc is 0x%08x, size is %d.\n", pDst, pSrc, size_out * dat_type_out_size); */
        if (pDst != pSrc)
            memcpy((void *)pDst, (void *)pSrc, size_out * dat_type_out_size);
        return err;
    }

    if ((line_stride_in > 0) || (surf_stride_in > 0))
    {
        input_addr_temp = (uint64_t)malloc(size_in * dat_type_in_size);
        if (dat_type_in != RINT8)
        {
            debug_info("Error, %s Wrong datatype!\n", __func__);
            return TYPE_ERR;
        }
        err = unvectorize((void *)pSrc, (void *)input_addr_temp, &cpu_parameters.cpu_operation_buffer.cast_buffers.src_data);
    }
    else
    {
        input_addr_temp = pSrc;
    }

    if ((line_stride_out > 0) || (surf_stride_out > 0))
    {
        if (dat_type_out != RINT8)
            return TYPE_ERR;
        output_addr_temp = (uint64_t)malloc(size_out * dat_type_out_size);
    }
    else
    {
        output_addr_temp = pDst;
    }
    for (h = 0; h < dat_h_in; h++)
    {
        for (w = 0; w < dat_w_in; w++)
        {
            for (c = 0; c < dat_c_in; c++)
            {
                uint64_t input_addr = input_addr_temp + (c + w * dat_c_in + h * dat_w_in * dat_c_in) * dat_type_in_size;
                uint64_t output_addr = output_addr_temp + (c + w * dat_c_out + h * dat_w_out * dat_c_out) * dat_type_out_size;
                if ((dat_type == RINT8) && (dat_type_in == RFLOAT))
                {
                    float_t input_data = *(volatile float *)input_addr / cpu_parameters.cpu_operation.cast_op.common.output_scale_factor[0];
                    int8_t dstint8p = (input_data >= 127.0) ? 127 : (input_data >= -128.0) ? (int8_t)input_data
                                                                                           : -128;
                    *(volatile int8_t *)output_addr = dstint8p;
                }
                else if ((dat_type == RFLOAT) && (dat_type_in == RINT8))
                {
                    float_t input_data = *(volatile int8_t *)input_addr * cpu_parameters.cpu_operation.cast_op.common.input_scale_factor[0];
                    *(volatile float *)output_addr = input_data;
                }
                else if ((dat_type == RFLOAT) && (dat_type_in == RINT))
                {
                    float_t input_data = (float)(*(volatile int32_t *)input_addr);
                    *(volatile float *)output_addr = input_data;
                }
                else
                {
                    debug_info("Error, In %s : %d, Unsurpport data type!\n", __func__, __LINE__);
                }
            }
        }
    }
    if ((line_stride_out > 0) || (surf_stride_out > 0))
    {
        err = vectorize((void *)output_addr_temp, (void *)pDst, &cpu_parameters.cpu_operation_buffer.cast_buffers.dst_data);
        free((void *)output_addr_temp);
    }
    if ((line_stride_in > 0) || (surf_stride_in > 0))
    {
        free((void *)input_addr_temp);
    }
    cpu_output_debug(cpu_parameters);
    // debug_info("\nExit %s;\n", __func__);
    // sifive_pl2cache0_flush((uintptr_t)pDst);
    return err;
}

int32_t executeTranspose(struct cpu_param cpu_parameters)
{
    // debug_info("Enter %s\n", __func__);
    int32_t err = 0;
    int32_t h, w, c;

    uint64_t output_addr_temp, input_addr_tmp;

    int32_t *axis = cpu_parameters.cpu_operation.transform_op.axis;

    uint64_t pDst = cpu_dev.task->address_list[cpu_parameters.cpu_operation_buffer.transform_op_buffers.dst_data.index];
    uint32_t dat_h_out = cpu_parameters.cpu_operation_buffer.transform_op_buffers.dst_data.height;
    uint32_t dat_w_out = cpu_parameters.cpu_operation_buffer.transform_op_buffers.dst_data.width;
    uint32_t dat_c_out = cpu_parameters.cpu_operation_buffer.transform_op_buffers.dst_data.channel;
    int32_t line_stride_out = cpu_parameters.cpu_operation_buffer.transform_op_buffers.dst_data.line_stride;
    int32_t surf_stride_out = cpu_parameters.cpu_operation_buffer.transform_op_buffers.dst_data.surf_stride;
    uint32_t dat_type_out = cpu_parameters.cpu_operation_buffer.transform_op_buffers.dst_data.datatype;

    uint32_t dat_type_in = cpu_parameters.cpu_operation_buffer.transform_op_buffers.src_data.datatype;

    uint32_t dat_type_out_size = (dat_type_out == RINT8) ? 1 : (dat_type_out == RBFLOAT) ? 2
                                                                                         : 4;
    uint32_t size_out = dat_h_out * dat_w_out * dat_c_out;

    uint64_t pSrc = cpu_dev.task->address_list[cpu_parameters.cpu_operation_buffer.transform_op_buffers.src_data.index];

    uint32_t dat_type_in_size = (dat_type_in == RINT8) ? 1 : (dat_type_in == RBFLOAT) ? 2
                                                                                      : 4;

    uint32_t dat_h_in = cpu_parameters.cpu_operation_buffer.transform_op_buffers.src_data.height;
    uint32_t dat_w_in = cpu_parameters.cpu_operation_buffer.transform_op_buffers.src_data.width;
    uint32_t dat_c_in = cpu_parameters.cpu_operation_buffer.transform_op_buffers.src_data.channel;
    int32_t line_stride_in = cpu_parameters.cpu_operation_buffer.transform_op_buffers.src_data.line_stride;
    int32_t surf_stride_in = cpu_parameters.cpu_operation_buffer.transform_op_buffers.src_data.surf_stride;

    uint32_t size_in = dat_h_in * dat_w_in * dat_c_in;
    if (size_in != size_out)
    {
        debug_info("\nError, input shape or output shape must be wrong!!!\n");
        return RESHAPE_ERR;
    }

    input_addr_tmp = pSrc;

    output_addr_temp = pDst;

    while ((axis[1] != 1) || (axis[2] != 2) || (axis[3] != 3))
    {
        if (axis[1] == 2)
        {
            for (h = 0; h < dat_h_in; h++)
            {
                for (w = 0; w < dat_w_in; w++)
                {
                    for (c = 0; c < dat_c_in; c++)
                    {
                        uint64_t input_addr = input_addr_tmp + (c + w * dat_c_in + h * dat_w_in * dat_c_in) * dat_type_in_size;
                        uint64_t output_addr = output_addr_temp + (c + h * dat_c_out + w * dat_w_out * dat_c_out) * dat_type_out_size;
                        if (dat_type_in == RINT8)
                        {
                            int8_t input_data = *(volatile int8_t *)input_addr;
                            *(volatile int8_t *)output_addr = input_data;
                        }
                        else if (dat_type_in == RFLOAT)
                        {
                            float_t input_data = *(volatile float_t *)input_addr;
                            *(volatile float_t *)output_addr = input_data;
                        }
                        else if (dat_type_in == RINT)
                        {
                            int32_t input_data = *(volatile int32_t *)input_addr;
                            *(volatile int32_t *)output_addr = input_data;
                        }
                        else if (dat_type_in == RUINT)
                        {
                            uint32_t input_data = *(volatile uint32_t *)input_addr;
                            *(volatile uint32_t *)output_addr = input_data;
                        }
                        else
                        {
                            debug_info("\nError,%s Unsupport data type\n", __func__);
                        }
                    }
                }
            }
            axis[1] = 1;
            if (axis[2] == 1)
                break;
            else
                axis[3] = 2;
        }
        else if (axis[1] == 3)
        {
            for (h = 0; h < dat_h_in; h++)
            {
                for (w = 0; w < dat_w_in; w++)
                {
                    for (c = 0; c < dat_c_in; c++)
                    {
                        uint64_t input_addr = input_addr_tmp + (c + w * dat_c_in + h * dat_w_in * dat_c_in) * dat_type_in_size;
                        uint64_t output_addr = output_addr_temp + (h + w * dat_c_out + c * dat_w_out * dat_c_out) * dat_type_out_size;
                        if (dat_type_in == RINT8)
                        {
                            int8_t input_data = *(volatile int8_t *)input_addr;
                            *(volatile int8_t *)output_addr = input_data;
                        }
                        else if (dat_type_in == RFLOAT)
                        {
                            float_t input_data = *(volatile float_t *)input_addr;
                            *(volatile float_t *)output_addr = input_data;
                        }
                        else if (dat_type_in == RINT)
                        {
                            int32_t input_data = *(volatile int32_t *)input_addr;
                            *(volatile int32_t *)output_addr = input_data;
                        }
                        else if (dat_type_in == RUINT)
                        {
                            uint32_t input_data = *(volatile uint32_t *)input_addr;
                            *(volatile uint32_t *)output_addr = input_data;
                        }
                        else
                        {
                            debug_info("\n%s Unsupport data type\n", __func__);
                        }
                    }
                }
            }
            axis[1] = 1;
            if (axis[3] == 1)
                break;
            else
                axis[2] = 3;
        }
        else if ((axis[2] == 3) && (axis[3] == 2))
        {
            for (h = 0; h < dat_h_in; h++)
            {
                for (w = 0; w < dat_w_in; w++)
                {
                    for (c = 0; c < dat_c_in; c++)
                    {
                        uint64_t input_addr = input_addr_tmp + (c + w * dat_c_in + h * dat_w_in * dat_c_in) * dat_type_in_size;
                        uint64_t output_addr = output_addr_temp + (w + c * dat_c_out + h * dat_w_out * dat_c_out) * dat_type_out_size;
                        if (dat_type_in == RINT8)
                        {
                            int8_t input_data = *(volatile int8_t *)input_addr;
                            *(volatile int8_t *)output_addr = input_data;
                        }
                        else if (dat_type_in == RFLOAT)
                        {
                            float_t input_data = *(volatile float_t *)input_addr;
                            *(volatile float_t *)output_addr = input_data;
                        }
                        else if (dat_type_in == RINT)
                        {
                            int32_t input_data = *(volatile int32_t *)input_addr;
                            *(volatile int32_t *)output_addr = input_data;
                        }
                        else if (dat_type_in == RUINT)
                        {
                            uint32_t input_data = *(volatile uint32_t *)input_addr;
                            *(volatile uint32_t *)output_addr = input_data;
                        }
                        else
                        {
                            debug_info("\n%s Unsupport data type\n", __func__);
                        }
                    }
                }
            }
            axis[2] = 2;
            axis[3] = 3;
            break;
        }
        else
        {
            debug_info("Warning in %s :%d, wrong axis, axis[1] = %d, axis[2] = %d, axis[3] = %d!\n", __func__, __LINE__,
                       axis[1], axis[2], axis[3]);
            /* return -1; */
        }
    }
    cpu_output_debug(cpu_parameters);
    // debug_info("\nExit %s;\n", __func__);
    return err;
}

int32_t executeTake(struct cpu_param cpu_parameters)
{
    // debug_info("Enter %s\n", __func__);
    int32_t err = 0;
    int32_t h, w, c;

    uint64_t output_addr_temp, input_addr_temp;
    uint64_t pDst = cpu_dev.task->address_list[cpu_parameters.cpu_operation_buffer.take_buffers.dst_data.index];
    uint32_t dat_h_out = cpu_parameters.cpu_operation_buffer.take_buffers.dst_data.height;
    uint32_t dat_w_out = cpu_parameters.cpu_operation_buffer.take_buffers.dst_data.width;
    uint32_t dat_c_out = cpu_parameters.cpu_operation_buffer.take_buffers.dst_data.channel;
    int32_t line_stride_out = cpu_parameters.cpu_operation_buffer.take_buffers.dst_data.line_stride;
    int32_t surf_stride_out = cpu_parameters.cpu_operation_buffer.take_buffers.dst_data.surf_stride;
    uint32_t dat_type_out = cpu_parameters.cpu_operation_buffer.take_buffers.dst_data.datatype;
    uint32_t dat_type_in = cpu_parameters.cpu_operation_buffer.take_buffers.src_data.datatype;
    uint32_t dat_type_out_size = (dat_type_out == RINT8) ? 1 : (dat_type_out == RBFLOAT) ? 2
                                                                                         : 4;
    uint32_t size_out = dat_h_out * dat_w_out * dat_c_out;

    uint64_t pSrc = cpu_dev.task->address_list[cpu_parameters.cpu_operation_buffer.take_buffers.src_data.index];
    uint32_t dat_type_in_size = (dat_type_in == RINT8) ? 1 : (dat_type_in == RBFLOAT) ? 2
                                                                                      : 4;
    uint32_t dat_h_in = cpu_parameters.cpu_operation_buffer.take_buffers.src_data.height;
    uint32_t dat_w_in = cpu_parameters.cpu_operation_buffer.take_buffers.src_data.width;
    uint32_t dat_c_in = cpu_parameters.cpu_operation_buffer.take_buffers.src_data.channel;
    int32_t line_stride_in = cpu_parameters.cpu_operation_buffer.take_buffers.src_data.line_stride;
    int32_t surf_stride_in = cpu_parameters.cpu_operation_buffer.take_buffers.src_data.surf_stride;
    uint32_t size_in = dat_h_in * dat_w_in * dat_c_in;

    uint64_t indic_pSrc = (uint64_t)cpu_dev.task->address_list[cpu_parameters.cpu_operation.take_op.indices.index];
    int32_t axis = cpu_parameters.cpu_operation.take_op.axis;
    uint32_t indic_h = cpu_parameters.cpu_operation.take_op.indices.height;
    uint32_t indic_w = cpu_parameters.cpu_operation.take_op.indices.width;
    uint32_t indic_c = cpu_parameters.cpu_operation.take_op.indices.channel;

    int32_t indic_line_stride_in = cpu_parameters.cpu_operation.take_op.indices.line_stride;
    int32_t indic_surf_stride_in = cpu_parameters.cpu_operation.take_op.indices.surf_stride;
    uint32_t indic_size_in = indic_h * indic_w * indic_c;
    uint32_t indic_dat_type_in = cpu_parameters.cpu_operation.take_op.indices.datatype;

    if (indic_dat_type_in != RINT)
    {
        debug_info("Error, %s : %d, indices type is wrong!\n");
        return TYPE_ERR;
    }

    if ((line_stride_in > 0) || (surf_stride_in > 0))
    {
        if (dat_type_in != RINT8)
        {
            debug_info("%s Wrong datatype!\n", __func__);
            return TYPE_ERR;
        }
        void *temp = (void *)malloc(size_in * dat_type_in_size);
        err = unvectorize((void *)pSrc, (void *)temp, &cpu_parameters.cpu_operation_buffer.take_buffers.src_data);
        input_addr_temp = (uint64_t)malloc(size_in * sizeof(float));
        for (h = 0; h < dat_h_in; h++)
        {
            for (w = 0; w < dat_w_in; w++)
            {
                for (c = 0; c < dat_c_in; c++)
                {
                    *(volatile float *)(input_addr_temp + (c + w * dat_c_in + h * dat_w_in * dat_c_in) * sizeof(float)) =
                        *(volatile int8_t *)((uint64_t)temp + (c + w * dat_c_in + h * dat_w_in * dat_c_in) * sizeof(int8_t)) *
                        cpu_parameters.cpu_operation.take_op.common.input_scale_factor[0];
                }
            }
        }
        free(temp);
    }
    else
    {
        input_addr_temp = pSrc;
    }

    if ((line_stride_out > 0) || (surf_stride_out > 0))
    {
        if (dat_type_out != RINT8)
            return TYPE_ERR;
        output_addr_temp = (uint64_t)malloc(size_out * dat_type_out_size);
    }
    else
    {
        output_addr_temp = pDst;
    }

    uint64_t indic_pSrc_temp = indic_pSrc;
    if (axis == 2)
    {
        for (w = 0; w < indic_size_in; w++)
        {
            int32_t w_temp = *(volatile int32_t *)(indic_pSrc_temp);
            indic_pSrc_temp += sizeof(int32_t);
            for (h = 0; h < dat_h_in; h++)
            {
                for (c = 0; c < dat_c_in; c++)
                {
                    uint64_t input_addr = input_addr_temp + (c + w_temp * dat_c_in + h * dat_w_in * dat_c_in) * dat_type_in_size;
                    uint64_t output_addr = output_addr_temp + (c + w * dat_c_out + h * dat_w_out * dat_c_out) * dat_type_out_size;
                    if (dat_type_out == RFLOAT)
                    {
                        float_t x = *(volatile float *)input_addr;
                        float_t y = (x);
                        *(volatile float_t *)output_addr = y;
                    }
                    else if (dat_type_out == RINT8)
                    {
                        float_t y = *(volatile float_t *)input_addr;
                        y /= cpu_parameters.cpu_operation.take_op.common.output_scale_factor[0];
                        int8_t dstint8p = (y >= 127.0) ? 127 : (y >= -128.0) ? (int8_t)y
                                                                             : -128;
                        *(volatile int8_t *)output_addr = dstint8p;
                    }
                }
            }
        }
    }
    if ((line_stride_out > 0) || (surf_stride_out > 0))
    {
        err = vectorize((void *)output_addr_temp, (void *)pDst, &cpu_parameters.cpu_operation_buffer.take_buffers.dst_data);
        free((void *)output_addr_temp);
    }
    if ((line_stride_in > 0) || (surf_stride_in > 0))
    {
        free((void *)input_addr_temp);
    }
    cpu_output_debug(cpu_parameters);
    // debug_info("\nExit %s;\n", __func__);
    // sifive_pl2cache0_flush((uintptr_t)pDst);
    return err;
}

int32_t executeOne_hot(struct cpu_param cpu_parameters)
{
    // debug_info("Enter %s\n", __func__);
    int32_t err = 0;
    int32_t h, w, c, i;

    uint64_t pDst = cpu_dev.task->address_list[cpu_parameters.cpu_operation_buffer.one_hot_buffers.dst_data.index];
    uint32_t dat_h_out = cpu_parameters.cpu_operation_buffer.one_hot_buffers.dst_data.height;
    uint32_t dat_w_out = cpu_parameters.cpu_operation_buffer.one_hot_buffers.dst_data.width;
    uint32_t dat_c_out = cpu_parameters.cpu_operation_buffer.one_hot_buffers.dst_data.channel;
    int32_t line_stride_out = cpu_parameters.cpu_operation_buffer.one_hot_buffers.dst_data.line_stride;
    int32_t surf_stride_out = cpu_parameters.cpu_operation_buffer.one_hot_buffers.dst_data.surf_stride;
    uint32_t dat_type_out = cpu_parameters.cpu_operation_buffer.one_hot_buffers.dst_data.datatype;
    uint32_t dat_type_in = cpu_parameters.cpu_operation_buffer.one_hot_buffers.src_data.datatype;
    uint32_t dat_type_out_size = (dat_type_out == RINT8) ? 1 : (dat_type_out == RBFLOAT) ? 2
                                                                                         : 4;
    uint32_t size_out = dat_h_out * dat_w_out * dat_c_out;

    uint64_t pSrc = cpu_dev.task->address_list[cpu_parameters.cpu_operation_buffer.one_hot_buffers.src_data.index];
    uint32_t dat_type_in_size = (dat_type_in == RINT8) ? 1 : (dat_type_in == RBFLOAT) ? 2
                                                                                      : 4;
    uint32_t dat_h_in = cpu_parameters.cpu_operation_buffer.one_hot_buffers.src_data.height;
    uint32_t dat_w_in = cpu_parameters.cpu_operation_buffer.one_hot_buffers.src_data.width;
    uint32_t dat_c_in = cpu_parameters.cpu_operation_buffer.one_hot_buffers.src_data.channel;
    int32_t line_stride_in = cpu_parameters.cpu_operation_buffer.one_hot_buffers.src_data.line_stride;
    int32_t surf_stride_in = cpu_parameters.cpu_operation_buffer.one_hot_buffers.src_data.surf_stride;
    uint32_t size_in = dat_h_in * dat_w_in * dat_c_in;

    int32_t depth = cpu_parameters.cpu_operation.one_hot_op.depth;
    float on_val = cpu_parameters.cpu_operation.one_hot_op.on_value;
    float off_val = cpu_parameters.cpu_operation.one_hot_op.off_value;
    // attention, input must be int32, output must be float
    if ((dat_type_in != RINT) || (dat_type_out != RFLOAT))
    {
        debug_info("In %s : %d, Data type Error!\n");
        return TYPE_ERR;
    }
    for (h = 0; h < dat_h_in; h++)
    {
        for (w = 0; w < dat_w_in; w++)
        {
            for (c = 0; c < dat_c_in; c++)
            {
                uint64_t input_addr = pSrc + (c + w * dat_c_in + h * dat_w_in * dat_c_in) * dat_type_in_size;
                uint64_t output_addr = pDst + (c + w * dat_c_in + h * dat_w_in * dat_c_in) * dat_type_out_size * depth;
                for (i = 0; i < depth; i++)
                {
                    if (i == *(volatile int32_t *)input_addr)
                    {
                        *(volatile float *)output_addr = on_val;
                    }
                    else
                    {
                        *(volatile float *)output_addr = off_val;
                    }
                    output_addr += sizeof(float);
                }
            }
        }
    }
    cpu_output_debug(cpu_parameters);
    // debug_info("\nExit %s;\n", __func__);
    // sifive_pl2cache0_flush((uintptr_t)pDst);
    return err;
}

int32_t executeExpend_dims(struct cpu_param cpu_parameters)
{
    // debug_info("Enter %s\n", __func__);
    int32_t err = 0;
    int32_t h, w, c;

    uint64_t output_addr_temp, input_addr_temp;
    uint64_t pDst = cpu_dev.task->address_list[cpu_parameters.cpu_operation_buffer.expand_dims_buffers.dst_data.index];
    uint32_t dat_h_out = cpu_parameters.cpu_operation_buffer.expand_dims_buffers.dst_data.height;
    uint32_t dat_w_out = cpu_parameters.cpu_operation_buffer.expand_dims_buffers.dst_data.width;
    uint32_t dat_c_out = cpu_parameters.cpu_operation_buffer.expand_dims_buffers.dst_data.channel;
    int32_t line_stride_out = cpu_parameters.cpu_operation_buffer.expand_dims_buffers.dst_data.line_stride;
    int32_t surf_stride_out = cpu_parameters.cpu_operation_buffer.expand_dims_buffers.dst_data.surf_stride;
    uint32_t dat_type_out = cpu_parameters.cpu_operation_buffer.expand_dims_buffers.dst_data.datatype;
    uint32_t dat_type_in = cpu_parameters.cpu_operation_buffer.expand_dims_buffers.src_data.datatype;
    uint32_t dat_type_out_size = (dat_type_out == RINT8) ? 1 : (dat_type_out == RBFLOAT) ? 2
                                                                                         : 4;
    uint32_t size_out = dat_h_out * dat_w_out * dat_c_out;

    uint64_t pSrc = cpu_dev.task->address_list[cpu_parameters.cpu_operation_buffer.expand_dims_buffers.src_data.index];
    uint32_t dat_type_in_size = (dat_type_in == RINT8) ? 1 : (dat_type_in == RBFLOAT) ? 2
                                                                                      : 4;
    uint32_t dat_h_in = cpu_parameters.cpu_operation_buffer.expand_dims_buffers.src_data.height;
    uint32_t dat_w_in = cpu_parameters.cpu_operation_buffer.expand_dims_buffers.src_data.width;
    uint32_t dat_c_in = cpu_parameters.cpu_operation_buffer.expand_dims_buffers.src_data.channel;
    int32_t line_stride_in = cpu_parameters.cpu_operation_buffer.expand_dims_buffers.src_data.line_stride;
    int32_t surf_stride_in = cpu_parameters.cpu_operation_buffer.expand_dims_buffers.src_data.surf_stride;
    uint32_t size_in = dat_h_in * dat_w_in * dat_c_in;

    if (size_in != size_out)
    {
        debug_info("\nError, Input shape or output shape must be wrong!!!\n");
        return RESHAPE_ERR;
    }

    if ((dat_type_in == dat_type_out) && (dat_type_in != RINT8))
    {
        /* debug_info("pDst is 0x%08x, pSrc is 0x%08x, size is %d.\n", pDst, pSrc, size_out * dat_type_out_size); */
        if (pDst != pSrc)
            memcpy((void *)pDst, (void *)pSrc, size_out * dat_type_out_size);
        return err;
    }

    if ((line_stride_in > 0) || (surf_stride_in > 0))
    {
        input_addr_temp = (uint64_t)malloc(size_in * dat_type_in_size);
        if (dat_type_in != RINT8)
        {
            debug_info("Error, %s Wrong datatype!\n", __func__);
            return TYPE_ERR;
        }
        err = unvectorize((void *)pSrc, (void *)input_addr_temp, &cpu_parameters.cpu_operation_buffer.expand_dims_buffers.src_data);
    }
    else
    {
        input_addr_temp = pSrc;
    }

    if ((line_stride_out > 0) || (surf_stride_out > 0))
    {
        if (dat_type_out != RINT8)
            return TYPE_ERR;
        output_addr_temp = (uint64_t)malloc(size_out * dat_type_out_size);
    }
    else
    {
        output_addr_temp = pDst;
    }

    for (h = 0; h < dat_h_in; h++)
    {
        for (w = 0; w < dat_w_in; w++)
        {
            for (c = 0; c < dat_c_in; c++)
            {
                uint64_t input_addr = input_addr_temp + (c + w * dat_c_in + h * dat_w_in * dat_c_in) * dat_type_in_size;
                uint64_t output_addr = output_addr_temp + (c + w * dat_c_in + h * dat_w_in * dat_c_in) * dat_type_out_size;
                if ((dat_type_in == RINT8) && (dat_type_out == RFLOAT))
                {
                    float_t x = *(volatile int8_t *)input_addr * cpu_parameters.cpu_operation.expand_dims_op.common.input_scale_factor[0];
                    float_t y = (x);
                    *(volatile float_t *)output_addr = y;
                }
                else if ((dat_type_in == RFLOAT) && (dat_type_out == RINT8))
                {
                    float_t x = *(volatile float_t *)input_addr;
                    float_t y = (x);
                    y /= cpu_parameters.cpu_operation.expand_dims_op.common.output_scale_factor[0];
                    int8_t dstint8p = (y >= 127.0) ? 127 : (y >= -128.0) ? (int8_t)y
                                                                         : -128;
                    *(volatile int8_t *)output_addr = dstint8p;
                }
                else if ((dat_type_in == RINT8) && (dat_type_out == RINT8))
                {
                    *(volatile int8_t *)output_addr = *(volatile int8_t *)input_addr;
                }
                else
                {
                    *(volatile int32_t *)output_addr = *(volatile int32_t *)input_addr;
                }
            }
        }
    }
    if ((line_stride_out > 0) || (surf_stride_out > 0))
    {
        err = vectorize((void *)output_addr_temp, (void *)pDst, &cpu_parameters.cpu_operation_buffer.expand_dims_buffers.dst_data);
        free((void *)output_addr_temp);
    }
    if ((line_stride_in > 0) || (surf_stride_in > 0))
    {
        free((void *)input_addr_temp);
    }
    cpu_output_debug(cpu_parameters);
    // debug_info("\nExit %s;\n", __func__);
    // sifive_pl2cache0_flush((uintptr_t)pDst);
    return err;
}

int32_t executeDense(struct cpu_param cpu_parameters)
{
    uint32_t h, w, w0, w1, c, dat_tmp, err = 0;
    void *output_addr_temp;
    void *input_addr_temp0;
    void *input_addr_temp1;
    uint64_t pDst = cpu_dev.task->address_list[cpu_parameters.cpu_operation_buffer.with_weight_op_buffers.dst_data.index];
    uint32_t dat_h_out = cpu_parameters.cpu_operation_buffer.with_weight_op_buffers.dst_data.height;
    uint32_t dat_w_out = cpu_parameters.cpu_operation_buffer.with_weight_op_buffers.dst_data.width;
    uint32_t dat_c_out = cpu_parameters.cpu_operation_buffer.with_weight_op_buffers.dst_data.channel;
    int32_t line_stride_out = cpu_parameters.cpu_operation_buffer.with_weight_op_buffers.dst_data.line_stride;
    int32_t surf_stride_out = cpu_parameters.cpu_operation_buffer.with_weight_op_buffers.dst_data.surf_stride;
    uint32_t dat_type_out = cpu_parameters.cpu_operation_buffer.with_weight_op_buffers.dst_data.datatype;
    uint32_t dat_type_out_size = (dat_type_out == RINT8) ? 1 : (dat_type_out == RBFLOAT) ? 2
                                                                                         : 4;
    uint32_t size_out = dat_h_out * dat_w_out * dat_c_out;
    if ((line_stride_out > 0) || (surf_stride_out > 0))
    {
        if (dat_type_out != RINT8)
            return TYPE_ERR;
        output_addr_temp = (void *)malloc(size_out * dat_type_out_size);
    }
    else
    {
        output_addr_temp = (void *)pDst;
    }
    // debug_info("%s: line_stride_out is %d, surf_stride_out is %d, output_addr_temp is 0x%08x, dst_data.address is 0x%08x.\n",
    //            __func__, line_stride_out, surf_stride_out, output_addr_temp,
    //            cpu_parameters.cpu_operation_buffer.with_weight_op_buffers.dst_data.address);
    uint64_t pSrc0 = cpu_dev.task->address_list[cpu_parameters.cpu_operation_buffer.with_weight_op_buffers.src_data.index];
    uint64_t pSrc1 = cpu_dev.task->address_list[cpu_parameters.cpu_operation.with_weight_op.weight.index];
    uint32_t dat_type = cpu_parameters.cpu_operation_buffer.with_weight_op_buffers.src_data.datatype;
    uint32_t weight_type = cpu_parameters.cpu_operation.with_weight_op.weight.datatype;
    uint32_t dat_type_in_size = (dat_type == RINT8) ? 1 : (dat_type == RBFLOAT) ? 2
                                                                                : 4;
    uint32_t dat_h = cpu_parameters.cpu_operation_buffer.with_weight_op_buffers.src_data.height;
    uint32_t dat_w = cpu_parameters.cpu_operation_buffer.with_weight_op_buffers.src_data.width;
    uint32_t dat_c = cpu_parameters.cpu_operation_buffer.with_weight_op_buffers.src_data.channel;
    int32_t dat_line_stride = cpu_parameters.cpu_operation_buffer.with_weight_op_buffers.src_data.line_stride;
    int32_t dat_surf_stride = cpu_parameters.cpu_operation_buffer.with_weight_op_buffers.src_data.surf_stride;
    uint32_t dat_size = dat_h * dat_w * dat_c;

    uint32_t wt_h = cpu_parameters.cpu_operation.with_weight_op.weight.height;
    uint32_t wt_w = cpu_parameters.cpu_operation.with_weight_op.weight.width;
    uint32_t wt_c = cpu_parameters.cpu_operation.with_weight_op.weight.channel;
    int32_t wt_line_stride = cpu_parameters.cpu_operation.with_weight_op.weight.line_stride;
    int32_t wt_surf_stride = cpu_parameters.cpu_operation.with_weight_op.weight.surf_stride;
    uint32_t wt_size = wt_h * wt_w * wt_c;

    uint32_t dense_type = cpu_parameters.cpu_operation.with_weight_op.sub_op_type;
    if (wt_size == 0)
    {
        debug_info("\nError, %s: Data shape is different with weight!\n", __func__);
        return SHAPE_ERR;
    }

    if (dense_type > 2)
    {
        debug_info("\nError, %s: dense_type error!\n", __func__);
        return MULTIPLY_ERR;
    }

    if ((dat_line_stride > 0) || (dat_surf_stride > 0))
    {
        if (dat_type != RINT8)
        {
            debug_info("Error, %s Wrong datatype!\n", __func__);
            return TYPE_ERR;
        }
        void *temp0 = (void *)malloc(dat_size * dat_type_in_size);
        err = unvectorize((void *)pSrc0, (void *)temp0, &cpu_parameters.cpu_operation_buffer.with_weight_op_buffers.src_data);
        input_addr_temp0 = (void *)malloc(dat_size * sizeof(float));
        for (h = 0; h < dat_h; h++)
        {
            for (w = 0; w < dat_w; w++)
            {
                for (c = 0; c < dat_c; c++)
                {
                    *(volatile float *)((uint64_t)input_addr_temp0 + (c + w * dat_c + h * dat_w * dat_c) * sizeof(float)) =
                        *(volatile int8_t *)((uint64_t)temp0 + (c + w * dat_c + h * dat_w * dat_c) * sizeof(int8_t)) *
                        cpu_parameters.cpu_operation.with_weight_op.common.input_scale_factor[0];
                }
            }
        }
        free(temp0);
    }
    else
    {
        input_addr_temp0 = (void *)pSrc0;
    }
    // debug_info("%s: dat_line_stride is %d, dat_surf_stride is %d, input_addr_temp0 is 0x%08x.\n", __func__, dat_line_stride, dat_surf_stride, input_addr_temp0);

    if ((wt_line_stride > 0) || (wt_surf_stride > 0))
    {
        if (weight_type != RINT8)
        {
            debug_info("Error, %s Wrong datatype!\n", __func__);
            return TYPE_ERR;
        }
        void *temp1 = (void *)malloc(wt_size * dat_type_in_size);
        err = unvectorize((void *)pSrc1, (void *)temp1, &cpu_parameters.cpu_operation.with_weight_op.weight);
        input_addr_temp1 = (void *)malloc(wt_size * sizeof(float));
        for (h = 0; h < wt_h; h++)
        {
            for (w = 0; w < wt_w; w++)
            {
                for (c = 0; c < wt_c; c++)
                {
                    *(volatile float *)((uint64_t)input_addr_temp1 + (c + w * wt_c + h * wt_w * wt_c) * sizeof(float)) =
                        *(volatile int8_t *)((uint64_t)temp1 + (c + w * wt_c + h * wt_w * wt_c) * sizeof(int8_t)) *
                        cpu_parameters.cpu_operation.with_weight_op.common.input_scale_factor[1];
                }
            }
        }
        free(temp1);
    }
    else
    {
        input_addr_temp1 = (void *)pSrc1;
    }
    // debug_info("%s: wt_line_stride is %d, wt_surf_stride is %d, input_addr_temp1 is 0x%08x.\n", __func__, wt_line_stride, wt_surf_stride, input_addr_temp1);
    // debug_info("Dat type is %d, Weight type is %d, dense_type is %d.\n", dat_type, weight_type, dense_type);

    void *output_addr = (void *)output_addr_temp;
    // debug_info(" output_addr 0x%08x\n", output_addr);
    // attention, y = x * W', need to transpose W
    /* if ((dat_h != 1) || (wt_h != 1)) */
    /* debug_info("Warning, (h,w) should reshape to (1,1,h,w)!\n"); */
    for (h = 0; h < dat_h; h++)
    {
        for (w0 = 0; w0 < dat_w; w0++)
        {
            for (w1 = 0; w1 < wt_w; w1++)
            {
                float sum = 0.0;

                for (c = 0; c < dat_c; c++)
                {
                    float *input_addr0 = (float *)((uint64_t)input_addr_temp0 + (c + w0 * dat_c + h * dat_w * dat_c) * sizeof(float));
                    float *input_addr1 = (float *)((uint64_t)input_addr_temp1 + (c + w1 * wt_c + h * wt_w * wt_c) * sizeof(float));
                    sum += (*(volatile float *)input_addr0) * (*(volatile float *)input_addr1);
                }

                if (dat_type_out == RINT8)
                {
                    float y = sum / cpu_parameters.cpu_operation.with_weight_op.common.output_scale_factor[0];
                    int8_t dstint8p = (y >= 127.0)    ? 127
                                      : (y >= -128.0) ? (int8_t)y
                                                      : -128;
                    *(volatile int8_t *)output_addr = dstint8p;
                    output_addr = (void *)((uint64_t)output_addr + sizeof(int8_t));
                }
                else if (dat_type_out == RFLOAT)
                {
                    *(volatile float *)output_addr = sum;
                    output_addr = (void *)((uint64_t)output_addr + sizeof(float));
                }
                else
                {
                    debug_info("\nError, %s Unsupport data type\n", __func__);
                }
            }
        }
    }
    if ((line_stride_out > 0) || (surf_stride_out > 0))
    {
        err = vectorize((void *)output_addr_temp, (void *)pDst, &cpu_parameters.cpu_operation_buffer.with_weight_op_buffers.dst_data);
        free((void *)output_addr_temp);
    }
    if ((dat_line_stride > 0) || (dat_surf_stride > 0))
        free((void *)input_addr_temp0);
    if ((wt_line_stride > 0) || (wt_surf_stride > 0))
        free((void *)input_addr_temp1);
    // debug_info("\nExit %s;\n", __func__);
    cpu_output_debug(cpu_parameters);
    return err;
}

int32_t executeBatch_matmul(struct cpu_param cpu_parameters)
{
    return executeDense(cpu_parameters);
}

int32_t executeSqueeze(struct cpu_param cpu_parameters)
{
    // debug_info("Enter %s\n", __func__);
    int32_t err = 0;
    int32_t h, w, c;

    uint64_t output_addr_temp, input_addr_temp;
    uint64_t pDst = cpu_dev.task->address_list[cpu_parameters.cpu_operation_buffer.transform_op_buffers.dst_data.index];
    uint32_t dat_h_out = cpu_parameters.cpu_operation_buffer.transform_op_buffers.dst_data.height;
    uint32_t dat_w_out = cpu_parameters.cpu_operation_buffer.transform_op_buffers.dst_data.width;
    uint32_t dat_c_out = cpu_parameters.cpu_operation_buffer.transform_op_buffers.dst_data.channel;
    int32_t line_stride_out = cpu_parameters.cpu_operation_buffer.transform_op_buffers.dst_data.line_stride;
    int32_t surf_stride_out = cpu_parameters.cpu_operation_buffer.transform_op_buffers.dst_data.surf_stride;
    uint32_t dat_type_out = cpu_parameters.cpu_operation_buffer.transform_op_buffers.dst_data.datatype;
    uint32_t dat_type_in = cpu_parameters.cpu_operation_buffer.transform_op_buffers.src_data.datatype;
    uint32_t dat_type_out_size = (dat_type_out == RINT8) ? 1 : (dat_type_out == RBFLOAT) ? 2
                                                                                         : 4;
    uint32_t size_out = dat_h_out * dat_w_out * dat_c_out;

    uint64_t pSrc = cpu_dev.task->address_list[cpu_parameters.cpu_operation_buffer.transform_op_buffers.src_data.index];
    uint32_t dat_type_in_size = (dat_type_in == RINT8) ? 1 : (dat_type_in == RBFLOAT) ? 2
                                                                                      : 4;
    uint32_t dat_h_in = cpu_parameters.cpu_operation_buffer.transform_op_buffers.src_data.height;
    uint32_t dat_w_in = cpu_parameters.cpu_operation_buffer.transform_op_buffers.src_data.width;
    uint32_t dat_c_in = cpu_parameters.cpu_operation_buffer.transform_op_buffers.src_data.channel;
    int32_t line_stride_in = cpu_parameters.cpu_operation_buffer.transform_op_buffers.src_data.line_stride;
    int32_t surf_stride_in = cpu_parameters.cpu_operation_buffer.transform_op_buffers.src_data.surf_stride;
    uint32_t size_in = dat_h_in * dat_w_in * dat_c_in;

    if (size_in != size_out)
    {
        debug_info("\nError, Input shape or output shape must be wrong!!!\n");
        return RESHAPE_ERR;
    }

    if (dat_type_in == dat_type_out)
    {
        // debug_info("pDst is 0x%08x, pSrc is 0x%08x, size is %d.\n", pDst, pSrc, size_out * dat_type_out_size);
        if (pDst != pSrc)
            memcpy((void *)pDst, (void *)pSrc, size_out * dat_type_out_size);
        return err;
    }

    if ((line_stride_in > 0) || (surf_stride_in > 0))
    {
        input_addr_temp = (uint64_t)malloc(size_in * dat_type_in_size);
        if (dat_type_in != RINT8)
        {
            debug_info("Error, %s Wrong datatype!\n", __func__);
            return TYPE_ERR;
        }
        err = unvectorize((void *)pSrc, (void *)input_addr_temp, &cpu_parameters.cpu_operation_buffer.transform_op_buffers.src_data);
    }
    else
    {
        input_addr_temp = pSrc;
    }

    if ((line_stride_out > 0) || (surf_stride_out > 0))
    {
        if (dat_type_out != RINT8)
            return TYPE_ERR;
        output_addr_temp = (uint64_t)malloc(size_out * dat_type_out_size);
    }
    else
    {
        output_addr_temp = pDst;
    }

    for (h = 0; h < dat_h_in; h++)
    {
        for (w = 0; w < dat_w_in; w++)
        {
            for (c = 0; c < dat_c_in; c++)
            {
                uint64_t input_addr = input_addr_temp + (c + w * dat_c_in + h * dat_w_in * dat_c_in) * dat_type_in_size;
                uint64_t output_addr = output_addr_temp + (c + w * dat_c_out + h * dat_w_out * dat_c_out) * dat_type_out_size;
                if ((dat_type_in == RINT8) && (dat_type_out == RFLOAT))
                {
                    float_t x = *(volatile int8_t *)input_addr * cpu_parameters.cpu_operation.transform_op.common.input_scale_factor[0];
                    float_t y = (x);
                    *(volatile float_t *)output_addr = y;
                }
                else if ((dat_type_in == RFLOAT) && (dat_type_out == RINT8))
                {
                    float_t x = *(volatile float_t *)input_addr;
                    float_t y = (x);
                    y /= cpu_parameters.cpu_operation.transform_op.common.output_scale_factor[0];
                    int8_t dstint8p = (y >= 127.0) ? 127 : (y >= -128.0) ? (int8_t)y
                                                                         : -128;
                    *(volatile int8_t *)output_addr = dstint8p;
                }
            }
        }
    }
    if ((line_stride_out > 0) || (surf_stride_out > 0))
    {
        err = vectorize((void *)output_addr_temp, (void *)pDst, &cpu_parameters.cpu_operation_buffer.transform_op_buffers.dst_data);
        free((void *)output_addr_temp);
    }
    if ((line_stride_in > 0) || (surf_stride_in > 0))
    {
        free((void *)input_addr_temp);
    }
    cpu_output_debug(cpu_parameters);
    // debug_info("\nExit %s;\n", __func__);
    return err;
}

int32_t executeResize(struct cpu_param cpu_parameters)
{
    uint32_t err = 0;
    uint64_t output_addr_temp, input_addr_temp;
    uint32_t dat_h_out = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.height;
    uint32_t dat_w_out = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.width;
    uint32_t dat_c_out = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.channel;
    int32_t line_stride_out = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.line_stride;
    int32_t surf_stride_out = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.surf_stride;
    uint32_t dat_type_out = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.datatype;
    uint32_t dat_type_in = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.datatype;
    uint32_t dat_type_out_size = (dat_type_out == RINT8) ? 1 : (dat_type_out == RBFLOAT) ? 2
                                                                                         : 4;
    uint32_t size_out = dat_h_out * dat_w_out * dat_c_out;

    cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.address = cpu_dev.task->address_list[cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.index];
    cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.address = cpu_dev.task->address_list[cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.index];

    uint64_t pSrc = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.address;
    uint64_t pDst = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.address;
    uint32_t dat_type_in_size = (dat_type_in == RINT8) ? 1 : (dat_type_in == RBFLOAT) ? 2
                                                                                      : 4;
    uint32_t dat_h_in = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.height;
    uint32_t dat_w_in = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.width;
    uint32_t dat_c_in = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.channel;
    int32_t line_stride_in = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.line_stride;
    int32_t surf_stride_in = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.surf_stride;
    uint32_t size_in = dat_h_in * dat_w_in * dat_c_in;
    uint32_t scale = dat_h_out / dat_h_in;

    uint32_t h, w, c, sx, sy, surf;
    if (((line_stride_in > 0) || (surf_stride_in > 0)) &&
        ((line_stride_out > 0) || (surf_stride_out > 0)))
    {
        if ((dat_type_in != RINT8) || (dat_type_out != RINT8))
        {
            debug_info("\nError, %s datatype is Wrong\n", __func__);
            return TYPE_ERR;
        }

        for (surf = 0; surf < ((dat_c_in - 1) / C_ATM + 1); surf++)
        {
            for (h = 0; h < dat_h_in; h++)
            {
                for (w = 0; w < dat_w_in; w++)
                {
                    for (c = 0; c < C_ATM; c++)
                    {
                        if ((dat_type_in == RINT8) && (dat_type_out == RFLOAT))
                        {
                            uint64_t input_addr = pSrc + (c + w * C_ATM + h * line_stride_in + surf * surf_stride_in) * sizeof(int8_t);
                            uint64_t output_addr = pDst + (c + w * scale * C_ATM + h * scale * line_stride_out + surf * surf_stride_out) * sizeof(int8_t);

                            float_t x = *(volatile int8_t *)input_addr * cpu_parameters.cpu_operation.common_only_op.common.input_scale_factor[0];
                            for (sy = 0; sy < scale; sy++)
                            {
                                for (sx = 0; sx < scale; sx++)
                                {
                                    int8_t *output_addr_temp = (int8_t *)(output_addr + (sx * C_ATM + sy * line_stride_out) * sizeof(int8_t));
                                    float_t y = (x);
                                    *(volatile float_t *)output_addr_temp = y;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    else
    {
        // first, change the vectorized data to unvectorize
        if ((line_stride_in > 0) || (surf_stride_in > 0))
        {
            input_addr_temp = (uint64_t)malloc(size_in * dat_type_in_size);
            err = unvectorize((void *)pSrc, (void *)input_addr_temp, &cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data);
        }
        else
        {
            input_addr_temp = pSrc;
        }
        // add data process code here
        if ((line_stride_out > 0) || (surf_stride_out > 0))
        {
            if (dat_type_out != RINT8)
                return TYPE_ERR;
            output_addr_temp = (uint64_t)malloc(size_out * dat_type_out_size);
        }
        else
        {
            output_addr_temp = pDst;
        }
        // debug_info("\nUpsample inputdata after unvectorize\n");
        for (h = 0; h < dat_h_in; h++)
        {
            // debug_info("\nh = %d\n", h);
            for (w = 0; w < dat_w_in; w++)
            {
                for (c = 0; c < dat_c_in; c++)
                {
                    uint64_t input_addr = input_addr_temp + (c + w * dat_c_in + h * dat_w_in * dat_c_in) * dat_type_in_size;
                    uint64_t output_addr = output_addr_temp + (c + scale * w * dat_c_in + scale * h * scale * dat_w_in * dat_c_in) * dat_type_out_size;
                    if (dat_type_out == RINT8)
                    {
                        for (sy = 0; sy < scale; sy++)
                        {
                            for (sx = 0; sx < scale; sx++)
                            {
                                int8_t *output_addr_tmp = (int8_t *)(output_addr + (sx * dat_c_in + sy * scale * dat_w_in * dat_c_in) * sizeof(int8_t));
                                if (dat_type_in == RINT8)
                                    *(volatile int8_t *)output_addr_tmp = *(volatile int8_t *)input_addr;
                                else
                                {
                                    float_t y = *(volatile float_t *)input_addr / cpu_parameters.cpu_operation.common_only_op.common.output_scale_factor[0];
                                    int8_t dstint8p = (y >= 127.0) ? 127 : (y >= -128.0) ? (int8_t)y
                                                                                         : -128;
                                    *(volatile int8_t *)output_addr_tmp = dstint8p;
                                }
                            }
                        }
                    }
                    else if (dat_type_out == RFLOAT)
                    {
                        float_t input_data = 0.0;
                        if (dat_type_in == RINT8)
                            input_data = *(volatile int8_t *)input_addr * cpu_parameters.cpu_operation.common_only_op.common.input_scale_factor[0];
                        else
                            input_data = *(volatile float_t *)input_addr;

                        for (sy = 0; sy < scale; sy++)
                        {
                            for (sx = 0; sx < scale; sx++)
                            {
                                float_t *output_addr_tmp = (float_t *)(output_addr + (sx * dat_c_in + sy * scale * dat_w_in * dat_c_in) * sizeof(float_t));
                                *(volatile float *)output_addr_tmp = input_data;
                            }
                        }
                    }
                    else if (dat_type_out == RINT)
                    {
                        int32_t input_data = *(volatile int32_t *)input_addr;
                        for (sy = 0; sy < scale; sy++)
                        {
                            for (sx = 0; sx < scale; sx++)
                            {
                                int32_t *output_addr_tmp = (int32_t *)(output_addr + (sx * dat_c_in + sy * scale * dat_w_in * dat_c_in) * sizeof(int32_t));
                                *(volatile int32_t *)output_addr_tmp = input_data;
                            }
                        }
                    }
                    else
                    {
                        debug_info("Error, %s unsupport datatype!\n", __func__);
                        err = TYPE_ERR;
                        return err;
                    }
                }
            }
        }
        // if the consumer is DLA, vectorize the output
        if ((line_stride_out > 0) || (surf_stride_out > 0))
        {
            err = vectorize((void *)output_addr_temp, (void *)pDst, &cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data);
            free((void *)output_addr_temp);
        }
        if ((line_stride_in > 0) || (surf_stride_in > 0))
        {
            free((void *)input_addr_temp);
        }
    }
    // debug_info("\nExit %s;\n", __func__);
    // sifive_pl2cache0_flush((uintptr_t)pDst);
    cpu_output_debug(cpu_parameters);
    return err;
}

// reduce_op: sum, mean, max, min, all, any, argmax, argmin
int32_t executeReduce(struct cpu_param cpu_parameters)
{
    // debug_info("Enter %s\n", __func__);
    int32_t err = 0;
    int32_t h, w, c;

    uint64_t output_addr_temp, input_addr_temp;

    int32_t *axis = cpu_parameters.cpu_operation.reduce_op.axis;

    uint64_t pDst = cpu_dev.task->address_list[cpu_parameters.cpu_operation_buffer.reduce_buffers.dst_data.index];
    uint32_t dat_h_out = cpu_parameters.cpu_operation_buffer.reduce_buffers.dst_data.height;
    uint32_t dat_w_out = cpu_parameters.cpu_operation_buffer.reduce_buffers.dst_data.width;
    uint32_t dat_c_out = cpu_parameters.cpu_operation_buffer.reduce_buffers.dst_data.channel;
    int32_t line_stride_out = cpu_parameters.cpu_operation_buffer.reduce_buffers.dst_data.line_stride;
    int32_t surf_stride_out = cpu_parameters.cpu_operation_buffer.reduce_buffers.dst_data.surf_stride;
    uint32_t dat_type_out = cpu_parameters.cpu_operation_buffer.reduce_buffers.dst_data.datatype;

    uint32_t dat_type_in = cpu_parameters.cpu_operation_buffer.reduce_buffers.src_data.datatype;

    uint32_t dat_type_out_size = (dat_type_out == RINT8) ? 1 : (dat_type_out == RBFLOAT) ? 2
                                                                                         : 4;
    uint32_t size_out = dat_h_out * dat_w_out * dat_c_out;

    uint64_t pSrc = cpu_dev.task->address_list[cpu_parameters.cpu_operation_buffer.reduce_buffers.src_data.index];

    uint32_t dat_type_in_size = (dat_type_in == RINT8) ? 1 : (dat_type_in == RBFLOAT) ? 2
                                                                                      : 4;

    uint32_t dat_h_in = cpu_parameters.cpu_operation_buffer.reduce_buffers.src_data.height;
    uint32_t dat_w_in = cpu_parameters.cpu_operation_buffer.reduce_buffers.src_data.width;
    uint32_t dat_c_in = cpu_parameters.cpu_operation_buffer.reduce_buffers.src_data.channel;
    int32_t line_stride_in = cpu_parameters.cpu_operation_buffer.reduce_buffers.src_data.line_stride;
    int32_t surf_stride_in = cpu_parameters.cpu_operation_buffer.reduce_buffers.src_data.surf_stride;

    uint32_t size_in = dat_h_in * dat_w_in * dat_c_in;

    if ((line_stride_in > 0) || (surf_stride_in > 0))
    {
        if (dat_type_in != RINT8)
        {
            debug_info("Error, %s Wrong datatype!\n", __func__);
            return TYPE_ERR;
        }
        void *temp0 = (void *)malloc(size_in * dat_type_in_size);
        err = unvectorize((void *)pSrc, (void *)temp0, &cpu_parameters.cpu_operation_buffer.reduce_buffers.src_data);
        input_addr_temp = (uint64_t)malloc(size_in * sizeof(float));
        for (h = 0; h < dat_h_in; h++)
        {
            for (w = 0; w < dat_w_in; w++)
            {
                for (c = 0; c < dat_c_in; c++)
                {
                    *(volatile float *)(input_addr_temp + (c + w * dat_c_in + h * dat_w_in * dat_c_in) * sizeof(float)) =
                        *(volatile int8_t *)((uint64_t)temp0 + (c + w * dat_c_in + h * dat_w_in * dat_c_in) * sizeof(int8_t)) *
                        cpu_parameters.cpu_operation.reduce_op.common.input_scale_factor[0];
                }
            }
        }
        free(temp0);
    }
    else
    {
        input_addr_temp = pSrc;
    }

    if ((line_stride_out > 0) || (surf_stride_out > 0))
    {
        if (dat_type_out != RINT8)
            return TYPE_ERR;
        output_addr_temp = (uint64_t)malloc(size_out * dat_type_out_size);
    }
    else
    {
        output_addr_temp = pDst;
    }

    if (axis[1] == 1 && axis[2] == 2) // fixup mean op in riscv for resnet50
    {

        for (c = 0; c < dat_c_in; c++)
        {
            float sum = 0.0;
            float y = 0.0;
            uint64_t output_addr = output_addr_temp + c * dat_type_out_size;
            for (h = 0; h < dat_h_in; h++)
            {
                for (w = 0; w < dat_w_in; w++)
                {

                    if (dat_type_out == RINT8)
                    {
                        uint64_t input_addr = input_addr_temp + (c + w * dat_c_in + h * dat_w_in * dat_c_in) * sizeof(float);
                        sum += *(volatile float *)input_addr;
                    }
                }
            }

            if (cpu_parameters.cpu_operation.reduce_op.common.op_type == MEAN)
            {
                y = sum / dat_w_in / dat_h_in / cpu_parameters.cpu_operation.reduce_op.common.output_scale_factor[0];
                // y = sum / dat_w_in / dat_h_in;
            }
            y = (y >= 0) ? y + 0.5 : y - 0.5;
            int8_t dstint8p = (y >= 127.0) ? 127 : (y >= -128.0) ? (int8_t)y
                                                                 : -128;
            *(volatile int8_t *)output_addr = dstint8p;
        }
    }

    else if (axis[3] == 3)
    {
        for (h = 0; h < dat_h_in; h++)
        {
            for (w = 0; w < dat_w_in; w++)
            {
                float sum = 0.0;
                float y = 0.0;
                uint64_t output_addr = output_addr_temp + (w + h * dat_w_out) * dat_type_out_size;
                for (c = 0; c < dat_c_in; c++)
                {
                    uint64_t input_addr = input_addr_temp + (c + w * dat_c_in + h * dat_w_in * dat_c_in) * sizeof(float);
                    sum += *(volatile float *)input_addr;
                }

                if (dat_type_out == RINT8)
                {
                    if (cpu_parameters.cpu_operation.reduce_op.common.op_type == SUM)
                        y = sum / cpu_parameters.cpu_operation.reduce_op.common.output_scale_factor[0];
                    else if (cpu_parameters.cpu_operation.reduce_op.common.op_type == MEAN)
                        y = sum / dat_c_in / cpu_parameters.cpu_operation.reduce_op.common.output_scale_factor[0];
                    else
                        debug_info("Error, in %s : %d, op_type error!\n", __func__, __LINE__);
                    int8_t dstint8p = (y >= 127.0) ? 127 : (y >= -128.0) ? (int8_t)y
                                                                         : -128;
                    *(volatile int8_t *)output_addr = dstint8p;
                }
                else if (dat_type_out == RFLOAT)
                {
                    if (cpu_parameters.cpu_operation.reduce_op.common.op_type == SUM)
                        y = sum;
                    else if (cpu_parameters.cpu_operation.reduce_op.common.op_type == MEAN)
                        y = sum / dat_c_in;
                    else
                        debug_info("Error, in %s : %d, op_type error!\n", __func__, __LINE__);
                    *(volatile float *)output_addr = y;
                }
                else
                {
                    debug_info("\nError, %s Unsupport data type\n", __func__);
                }
            }
        }
    }
    else if (axis[2] == 2)
    {
        for (h = 0; h < dat_h_in; h++)
        {
            for (c = 0; c < dat_c_in; c++)
            {
                float sum = 0.0;
                float y = 0.0;
                uint64_t output_addr = output_addr_temp + (c + h * dat_c_out) * dat_type_out_size;
                for (w = 0; w < dat_w_in; w++)
                {
                    uint64_t input_addr = input_addr_temp + (c + w * dat_c_in + h * dat_w_in * dat_c_in) * sizeof(float);
                    sum += *(volatile float *)input_addr;
                }
                if (dat_type_out == RINT8)
                {
                    if (cpu_parameters.cpu_operation.reduce_op.common.op_type == SUM)
                        y = sum / cpu_parameters.cpu_operation.reduce_op.common.output_scale_factor[0];
                    else if (cpu_parameters.cpu_operation.reduce_op.common.op_type == MEAN)
                        y = sum / dat_w_in / cpu_parameters.cpu_operation.reduce_op.common.output_scale_factor[0];
                    else
                        debug_info("Error, in %s : %d, op_type error!\n", __func__, __LINE__);
                    int8_t dstint8p = (y >= 127.0) ? 127 : (y >= -128.0) ? (int8_t)y
                                                                         : -128;
                    *(volatile int8_t *)output_addr = dstint8p;
                }
                else if (dat_type_out == RFLOAT)
                {
                    if (cpu_parameters.cpu_operation.reduce_op.common.op_type == SUM)
                        y = sum;
                    else if (cpu_parameters.cpu_operation.reduce_op.common.op_type == MEAN)
                        y = sum / dat_c_in;
                    else
                        debug_info("Error, in %s : %d, op_type error!\n", __func__, __LINE__);
                    *(volatile float *)output_addr = y;
                }
                else
                {
                    debug_info("\n%s Unsupport data type\n", __func__);
                }
            }
        }
    }
    else if (axis[1] == 1)
    {
        for (w = 0; w < dat_w_in; w++)
        {
            for (c = 0; c < dat_c_in; c++)
            {
                float sum = 0.0;
                float y = 0.0;
                uint64_t output_addr = output_addr_temp + (c + w * dat_c_out) * dat_type_out_size;
                for (h = 0; h < dat_h_in; h++)
                {
                    uint64_t input_addr = input_addr_temp + (c + w * dat_c_in + h * dat_w_in * dat_c_in) * sizeof(float);
                    sum += *(volatile float *)input_addr;
                }
                if (dat_type_out == RINT8)
                {
                    if (cpu_parameters.cpu_operation.reduce_op.common.op_type == SUM)
                        y = sum / cpu_parameters.cpu_operation.reduce_op.common.output_scale_factor[0];
                    else if (cpu_parameters.cpu_operation.reduce_op.common.op_type == MEAN)
                        y = sum / dat_h_in / cpu_parameters.cpu_operation.reduce_op.common.output_scale_factor[0];
                    else
                        debug_info("Error, in %s : %d, op_type error!\n", __func__, __LINE__);
                    int8_t dstint8p = (y >= 127.0) ? 127 : (y >= -128.0) ? (int8_t)y
                                                                         : -128;
                    *(volatile int8_t *)output_addr = dstint8p;
                }
                else if (dat_type_out == RFLOAT)
                {
                    if (cpu_parameters.cpu_operation.reduce_op.common.op_type == SUM)
                        y = sum;
                    else if (cpu_parameters.cpu_operation.reduce_op.common.op_type == MEAN)
                        y = sum / dat_c_in;
                    else
                        debug_info("Error, in %s : %d, op_type error!\n", __func__, __LINE__);
                    *(volatile float *)output_addr = y;
                }
                else
                {
                    debug_info("\n%s Unsupport data type\n", __func__);
                }
            }
        }
    }
    else
    {
        debug_info("%s unsupported axis %d !\n", __func__, axis);
        err = AXIS_ERR;
    }

    if ((line_stride_out > 0) || (surf_stride_out > 0))
    {
        err = vectorize((void *)output_addr_temp, (void *)pDst, &cpu_parameters.cpu_operation_buffer.reduce_buffers.dst_data);
        free((void *)output_addr_temp);
    }

    if ((line_stride_in > 0) || (surf_stride_in > 0))
    {
        free((void *)input_addr_temp);
    }
    cpu_output_debug(cpu_parameters);
    // debug_info("\nExit %s;\n", __func__);
    return err;
}

int32_t executeSplit(struct cpu_param cpu_parameters)
{
    // debug_info("Enter %s\n", __func__);
    int32_t err = 0;
    int32_t h, w, c, i;

    uint64_t input_addr_temp, output_addr_temp;
    uint64_t pSrc = cpu_dev.task->address_list[cpu_parameters.cpu_operation_buffer.split_buffers.src_data.index];
    uint32_t dat_type_in = cpu_parameters.cpu_operation_buffer.split_buffers.src_data.datatype;
    uint32_t dat_type_in_size = (dat_type_in == RINT8) ? 1 : (dat_type_in == RBFLOAT) ? 2
                                                                                      : 4;
    uint32_t dat_h_in = cpu_parameters.cpu_operation_buffer.split_buffers.src_data.height;
    uint32_t dat_w_in = cpu_parameters.cpu_operation_buffer.split_buffers.src_data.width;
    uint32_t dat_c_in = cpu_parameters.cpu_operation_buffer.split_buffers.src_data.channel;
    int32_t line_stride_in = cpu_parameters.cpu_operation_buffer.split_buffers.src_data.line_stride;
    int32_t surf_stride_in = cpu_parameters.cpu_operation_buffer.split_buffers.src_data.surf_stride;
    uint32_t size_in = dat_h_in * dat_w_in * dat_c_in;

    uint32_t output_num = cpu_parameters.cpu_operation.split_op.common.output_num;
    uint32_t split_num = cpu_parameters.cpu_operation.split_op.indices[0];
    uint32_t axis = cpu_parameters.cpu_operation.split_op.axis;

    if (split_num != output_num)
    {
        debug_info("Error in %s : %d, parameter is Wrong!\n", __func__, __LINE__);
        return -1;
    }

    if ((line_stride_in > 0) || (surf_stride_in > 0))
    {
        input_addr_temp = (uint64_t)malloc(size_in * dat_type_in_size);
        if (dat_type_in != RINT8)
        {
            debug_info("Error, %s Wrong datatype!\n", __func__);
            return TYPE_ERR;
        }
        err = unvectorize((void *)pSrc, (void *)input_addr_temp, &cpu_parameters.cpu_operation_buffer.split_buffers.src_data);
    }
    else
    {
        input_addr_temp = pSrc;
    }

    for (i = 0; i < output_num; i++)
    {
        uint64_t pDst = cpu_dev.task->address_list[cpu_parameters.cpu_operation.split_op.dst_data[i].index];
        /* debug_info("Split output address[%d] is 0x%08x.\n", i, pDst); */
        uint32_t dat_h_out = cpu_parameters.cpu_operation.split_op.dst_data[i].height;
        uint32_t dat_w_out = cpu_parameters.cpu_operation.split_op.dst_data[i].width;
        uint32_t dat_c_out = cpu_parameters.cpu_operation.split_op.dst_data[i].channel;
        int32_t line_stride_out = cpu_parameters.cpu_operation.split_op.dst_data[i].line_stride;
        int32_t surf_stride_out = cpu_parameters.cpu_operation.split_op.dst_data[i].surf_stride;
        uint32_t dat_type_out = cpu_parameters.cpu_operation.split_op.dst_data[i].datatype;
        uint32_t dat_type_out_size = (dat_type_out == RINT8) ? 1 : (dat_type_out == RBFLOAT) ? 2
                                                                                             : 4;
        uint32_t size_out = dat_h_out * dat_w_out * dat_c_out;

        if ((line_stride_out > 0) || (surf_stride_out > 0))
        {
            if (dat_type_out != RINT8)
                return TYPE_ERR;
            output_addr_temp = (uint64_t)malloc(size_out * dat_type_out_size);
        }
        else
        {
            output_addr_temp = pDst;
        }

        uint64_t input_addr = 0;
        if (dat_type_in != dat_type_out)
        {
            debug_info("Error in %s : %d, parameter is Wrong!\n", __func__, __LINE__);
            return -1;
        }
        for (h = 0; h < dat_h_out; h++)
        {
            for (w = 0; w < dat_w_out; w++)
            {
                for (c = 0; c < dat_c_out; c++)
                {
                    if (axis == 1)
                        input_addr = input_addr_temp + (c + w * dat_c_in + (h + i * dat_h_out) * dat_w_in * dat_c_in) * dat_type_in_size;
                    else if (axis == 2)
                        input_addr = input_addr_temp + (c + (w + i * dat_w_out) * dat_c_in + h * dat_w_in * dat_c_in) * dat_type_in_size;
                    else if (axis == 3)
                        input_addr = input_addr_temp + ((c + i * dat_c_out) + w * dat_c_in + h * dat_w_in * dat_c_in) * dat_type_in_size;
                    else
                    {
                        debug_info("Error in %s : %d, parameter is Wrong!\n", __func__, __LINE__);
                        return -1;
                    }
                    uint64_t output_addr = output_addr_temp + (c + w * dat_c_out + h * dat_w_out * dat_c_out) * dat_type_out_size;
                    if ((dat_type_in == RINT8) && (dat_type_out == RINT8))
                    {
                        float_t x = *(volatile int8_t *)input_addr * cpu_parameters.cpu_operation.split_op.common.input_scale_factor[0];
                        float_t y = x / cpu_parameters.cpu_operation.split_op.common.output_scale_factor[i];
                        int8_t dstint8p = (y >= 127.0) ? 127 : (y >= -128.0) ? (int8_t)y
                                                                             : -128;
                        *(volatile int8_t *)output_addr = dstint8p;
                    }
                    else if ((dat_type_in == RINT) && (dat_type_out == RINT))
                        *(volatile int32_t *)output_addr = *(volatile int32_t *)input_addr;
                    else if ((dat_type_in == RFLOAT) && (dat_type_out == RFLOAT))
                        *(volatile float *)output_addr = *(volatile float *)input_addr;
                    else if ((dat_type_in == RINT8) && (dat_type_out == RFLOAT))
                        *(volatile float *)output_addr = *(volatile int8_t *)input_addr * cpu_parameters.cpu_operation.split_op.common.input_scale_factor[0];
                    else if ((dat_type_in == RFLOAT) && (dat_type_out == RINT8))
                    {
                        float_t x = *(volatile float_t *)input_addr;
                        float_t y = x / cpu_parameters.cpu_operation.split_op.common.output_scale_factor[i];
                        int8_t dstint8p = (y >= 127.0) ? 127 : (y >= -128.0) ? (int8_t)y
                                                                             : -128;
                        *(volatile int8_t *)output_addr = dstint8p;
                    }
                }
            }
        }
        if ((line_stride_out > 0) || (surf_stride_out > 0))
        {
            err = vectorize((void *)output_addr_temp, (void *)pDst, &cpu_parameters.cpu_operation.split_op.dst_data[i]);
            free((void *)output_addr_temp);
        }
    }
    if ((line_stride_in > 0) || (surf_stride_in > 0))
    {
        free((void *)input_addr_temp);
    }
    cpu_output_debug(cpu_parameters);
    // debug_info("\nExit %s;\n", __func__);
    return err;
}

int32_t executePool2D(struct cpu_param cpu_parameters)
{
    // debug_info("Enter %s\n", __func__);
    int32_t err = 0;
    int32_t h, w, c;

    uint64_t output_addr_temp, input_addr_temp;
    uint64_t pDst = cpu_dev.task->address_list[cpu_parameters.cpu_operation_buffer.transform_op_buffers.dst_data.index];
    uint32_t dat_h_out = cpu_parameters.cpu_operation_buffer.transform_op_buffers.dst_data.height;
    uint32_t dat_w_out = cpu_parameters.cpu_operation_buffer.transform_op_buffers.dst_data.width;
    uint32_t dat_c_out = cpu_parameters.cpu_operation_buffer.transform_op_buffers.dst_data.channel;
    int32_t line_stride_out = cpu_parameters.cpu_operation_buffer.transform_op_buffers.dst_data.line_stride;
    int32_t surf_stride_out = cpu_parameters.cpu_operation_buffer.transform_op_buffers.dst_data.surf_stride;
    uint32_t dat_type_out = cpu_parameters.cpu_operation_buffer.transform_op_buffers.dst_data.datatype;
    uint32_t dat_type_in = cpu_parameters.cpu_operation_buffer.transform_op_buffers.src_data.datatype;
    uint32_t dat_type_out_size = (dat_type_out == RINT8) ? 1 : (dat_type_out == RBFLOAT) ? 2
                                                                                         : 4;
    uint32_t size_out = dat_h_out * dat_w_out * dat_c_out;

    uint64_t pSrc = cpu_dev.task->address_list[cpu_parameters.cpu_operation_buffer.transform_op_buffers.src_data.index];
    uint32_t dat_type_in_size = (dat_type_in == RINT8) ? 1 : (dat_type_in == RBFLOAT) ? 2
                                                                                      : 4;
    uint32_t dat_h_in = cpu_parameters.cpu_operation_buffer.transform_op_buffers.src_data.height;
    uint32_t dat_w_in = cpu_parameters.cpu_operation_buffer.transform_op_buffers.src_data.width;
    uint32_t dat_c_in = cpu_parameters.cpu_operation_buffer.transform_op_buffers.src_data.channel;
    int32_t line_stride_in = cpu_parameters.cpu_operation_buffer.transform_op_buffers.src_data.line_stride;
    int32_t surf_stride_in = cpu_parameters.cpu_operation_buffer.transform_op_buffers.src_data.surf_stride;
    uint32_t size_in = dat_h_in * dat_w_in * dat_c_in;

    uint32_t pool_type = cpu_parameters.cpu_operation.pool2d_op.pool_type;

    uint32_t kernel_height = cpu_parameters.cpu_operation.pool2d_op.kernel[0];
    uint32_t kernel_width = cpu_parameters.cpu_operation.pool2d_op.kernel[1];

    uint32_t strides_height = cpu_parameters.cpu_operation.pool2d_op.strides[0];
    uint32_t strides_width = cpu_parameters.cpu_operation.pool2d_op.strides[1];

    uint32_t padding_top = cpu_parameters.cpu_operation.pool2d_op.padding[0];
    uint32_t padding_left = cpu_parameters.cpu_operation.pool2d_op.padding[1];
    uint32_t padding_bottom = cpu_parameters.cpu_operation.pool2d_op.padding[2];
    uint32_t padding_right = cpu_parameters.cpu_operation.pool2d_op.padding[3];

    if ((line_stride_in > 0) || (surf_stride_in > 0))
    {
        if (dat_type_in != RINT8)
        {
            debug_info("%s Wrong datatype!\n", __func__);
            return TYPE_ERR;
        }
        void *temp0 = (void *)malloc(size_in * dat_type_in_size);
        err = unvectorize((void *)pSrc, (void *)temp0, &cpu_parameters.cpu_operation_buffer.reduce_buffers.src_data);
        input_addr_temp = (uint64_t)malloc(size_in * sizeof(float));
        for (h = 0; h < dat_h_in; h++)
        {
            for (w = 0; w < dat_w_in; w++)
            {
                for (c = 0; c < dat_c_in; c++)
                {
                    *(volatile float *)(input_addr_temp + (c + w * dat_c_in + h * dat_w_in * dat_c_in) * sizeof(float)) =
                        *(volatile int8_t *)((uint64_t)temp0 + (c + w * dat_c_in + h * dat_w_in * dat_c_in) * sizeof(int8_t)) *
                        cpu_parameters.cpu_operation.reduce_op.common.input_scale_factor[0];
                }
            }
        }
        free(temp0);
    }
    else
    {
        input_addr_temp = pSrc;
    }

    if ((line_stride_out > 0) || (surf_stride_out > 0))
    {
        if (dat_type_out != RINT8)
            return TYPE_ERR;
        output_addr_temp = (uint64_t)malloc(size_out * dat_type_out_size);
    }
    else
    {
        output_addr_temp = pDst;
    }

    for (uint32_t c = 0; c < dat_c_out; c++)
    {
        for (uint32_t h = 0; h < dat_h_out; h++)
        {
            for (uint32_t w = 0; w < dat_w_out; w++)
            {

                int h_start = h * strides_height - padding_top;
                int h_end = h_start + kernel_height;
                if (h_end > dat_h_in + padding_bottom)
                    h_end = dat_h_in + padding_bottom;
                int w_start = w * strides_width - padding_left;
                int w_end = w_start + kernel_width;
                if (w_end > dat_w_in + padding_right)
                    w_end = dat_w_in + padding_right;

                int pool_size = (h_end - h_start) * (w_end - w_start);

                h_start = h_start > 0 ? h_start : 0;
                w_start = w_start > 0 ? w_start : 0;
                h_end = h_end < dat_h_in ? h_end : dat_h_in;
                w_end = w_end < dat_w_in ? w_end : dat_w_in;

                uint64_t output_addr_offset = (c + w * dat_c_out + h * dat_w_out * dat_c_out) * dat_type_out_size;
                float *max_addr = (float *)(input_addr_temp + (h_start * dat_c_in * dat_w_in + w_start * dat_c_in + c) * sizeof(float));
                float max = *(volatile float *)max_addr;
                float sum;
                float *tmp_addr;
                if (pool_type == 0)
                {
                    for (int i = h_start; i < h_end; i++)
                    {
                        for (int j = w_start; j < w_end; j++)
                        {
                            tmp_addr = (float *)(input_addr_temp + (i * dat_c_in * dat_w_in + j * dat_c_in + c) * sizeof(float));
                            float tmp = *(volatile float *)tmp_addr;
                            max = max > tmp ? max : tmp;
                        }
                    }
                    if (dat_type_out == RINT8)
                    {
                        max /= cpu_parameters.cpu_operation.common_only_op.common.output_scale_factor[0];
                        max = (max >= 0) ? max + 0.5 : max - 0.5;
                        int8_t dstint8p = (max >= 127.0) ? 127 : (max >= -128.0) ? (int8_t)max
                                                                                 : -128;

                        *(volatile int8_t *)(output_addr_temp + output_addr_offset) = dstint8p;
                    }
                    if (dat_type_out == RFLOAT)
                    {
                        *(volatile float *)(output_addr_temp + output_addr_offset) = max;
                    }
                }
                else if (pool_type == 1)
                {
                    for (int i = h_start; i < h_end; i++)
                    {
                        for (int j = w_start; j < w_end; j++)
                        {
                            tmp_addr = (float *)(input_addr_temp + (i * dat_c_in * dat_w_in + j * dat_c_in + c) * sizeof(float));
                            float tmp = *(volatile float *)tmp_addr;
                            sum += tmp;
                        }
                    }

                   sum = sum / pool_size;

                    if (dat_type_out == RINT8)
                    {
                        sum /= cpu_parameters.cpu_operation.common_only_op.common.output_scale_factor[0];
                        sum = (sum >= 0) ? sum + 0.5 : sum - 0.5;
                        int8_t dstint8p = (sum >= 127.0) ? 127 : (sum >= -128.0) ? (int8_t)sum
                                                                                 : -128;

                        *(volatile int8_t *)(output_addr_temp + output_addr_offset) = dstint8p;
                    }
                    if (dat_type_out == RFLOAT)
                    {
                        *(volatile float *)(output_addr_temp + output_addr_offset) = sum;
                    }
                }
            }
        }
    }

    if ((line_stride_out > 0) || (surf_stride_out > 0))
    {
        err = vectorize((void *)output_addr_temp, (void *)pDst, &cpu_parameters.cpu_operation_buffer.transform_op_buffers.dst_data);
        free((void *)output_addr_temp);
    }
    if ((line_stride_in > 0) || (surf_stride_in > 0))
    {
        free((void *)input_addr_temp);
    }
    cpu_output_debug(cpu_parameters);
    // debug_info("\nExit %s;\n", __func__);
    return err;
}
