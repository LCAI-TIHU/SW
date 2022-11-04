#include <stdarg.h>
#include <string.h>
#include "device_init.h"
#include "dla_debug.h"
#include "operations.h"


static struct cpu_param cpu_parameters;

int32_t read_cpu_address(struct cpu_device *cpu_dev, int16_t index, void *dst)
{
    uint64_t *temp = (uint64_t *)dst;
    struct cpu_task *task = (struct cpu_task *)cpu_dev->task;
    uint64_t *addr_list = (uint64_t *)task->address_list;
    if (index == -1 || index > task->num_addresses)
        return -EINVAL;

    *temp = addr_list[index];
    return 0;
}

int32_t cpu_data_read(struct cpu_device *cpu_dev,
                      uint64_t src, void *dst,
                      uint32_t size, uint64_t offset)
{

//  debug_trace("CPU: Read data from address: 0x%x, to address:0x%x, len:%d\n",src, dst, size);
    memcpy(dst, (void *)(src+offset), size);
    /*
        int32_t i;
        for(i = 0; i < size; i++){
            *(char *)(dst+i) = *(char *)(src+offset+i);
        }
    */
    return 0;

}

int32_t cpu_task_submit(struct cpu_device *cpu_dev, struct cpu_task *task)
{
    /* debug_info("Enter %s\n", __func__); */
    int32_t err = 0;
    uint32_t task_complete = 0;

    uint8_t op_type = task->cpu_task_pt->cpu_parameters.cpu_operation.common_only_op.common.op_type;


    if((op_type == SUM)|| (op_type == MEAN))
    {
        /* debug_info("MEAN operation\n"); */
        err = executeReduce(task->cpu_task_pt->cpu_parameters);
    }
    else if((op_type == ADD)||(op_type  == SUBTRACT)||(op_type  == MULTIPLY) || (op_type  == DIVIDE)||(op_type  == POWER))
    {
        /* debug_info("AddV2_Sub_Mult_Div_Power operation\n"); */
        err = executeAddV2_Sub_Mult_Div_Power(task->cpu_task_pt->cpu_parameters);
    }
    else  if(op_type  == DENSE)
    {
        /* debug_info("DENSE operation\n"); */
        err = executeDense(task->cpu_task_pt->cpu_parameters);
    }
    else if(op_type == SOFTMAX)
    {
        /* debug_info("Softmax operation\n"); */
        err = executeSoftmax(task->cpu_task_pt->cpu_parameters);
    }
    else if(op_type == RESHAPE)
    {
    	/* debug_info("Reshape operation\n"); */
        err = executeReshape(task->cpu_task_pt->cpu_parameters);
    }

    else if(op_type == RESIZE)  //upsample == resize
    {
        /* debug_info("Resize operation\n"); */
       
        err = executeResize(task->cpu_task_pt->cpu_parameters);
        //err = executeUpsampe_nearest(task->cpu_task_pt->cpu_parameters);
    }
    else if(op_type == CONCAT)
    {
        /* debug_info("Concat operation\n"); */
        err = executeConcat(task->cpu_task_pt->cpu_parameters);
    }
    else if(op_type  == SIGMOID)
    {
        /* debug_info("SIGMOID operation\n"); */
        err = executeSigmoid(task->cpu_task_pt->cpu_parameters);
    }

    else if(op_type == STRIDED_SLICE )
    {
        /* debug_info("Slice operation\n"); */
        err = executeSlice(task->cpu_task_pt->cpu_parameters);
    }

    else if(op_type == EXP||(op_type  == TANH))
    {
        /* debug_info("Expf operation\n"); */
        err = executeExpf_Tanh(task->cpu_task_pt->cpu_parameters);
    }
    else if(op_type  == EXPAND_DIMS)
    {
        /* debug_info("EXPAND_DIMS operation\n"); */
        err = executeExpend_dims(task->cpu_task_pt->cpu_parameters);
    }
    else if(op_type  == TAKE)
    {
        /* debug_info("TAKE operation\n"); */
        err = executeTake(task->cpu_task_pt->cpu_parameters);
    }
    else if(op_type  == ONE_HOT)
    {
        /* debug_info("ONE_HOT operation\n"); */
        err = executeOne_hot(task->cpu_task_pt->cpu_parameters);
    }
    else if(op_type  == BATCH_MATMUL)
    {
        /* debug_info("BATCH_MATMUL operation\n"); */
        err = executeBatch_matmul(task->cpu_task_pt->cpu_parameters);
    }
    else if(op_type  == TRANSPOSE)
    {
        /* debug_info("TRANSPOSE operation\n"); */
        err = executeTranspose(task->cpu_task_pt->cpu_parameters);
    }
    else if(op_type  == CAST)
    {
        /* debug_info("CAST operation\n"); */
        err = executeCast(task->cpu_task_pt->cpu_parameters);
    }
    else if(op_type  == SQUEEZE)
    {
        /* debug_info("SQUEEZE operation\n"); */
        err = executeSqueeze(task->cpu_task_pt->cpu_parameters);
    }
    else if(op_type  == SPLIT)
    {
        /* debug_info("SPLIT operation\n"); */
        err = executeSplit(task->cpu_task_pt->cpu_parameters);
    }
    else if(op_type  == POOL2D)
    {
        /* debug_info("POOL2D operation\n"); */
        err = executePool2D(task->cpu_task_pt->cpu_parameters);
    }
    else
    {
        err = NO_CPU_TASK;
    }

    if (err)
    {
        debug_info("Task execution failed, err = %d\n",err);
        return err;
    }
exit:
    //debug_info("Exit %s\n", __func__);
    return err;
}
