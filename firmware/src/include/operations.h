#ifndef __OPERATIONS_H
#define __OPERATIONS_H
#include <stdint.h>
#include <device_init.h>
#include "dla_debug.h"
// #include "riscv_vector.h" 

//#define __MATH_VECTOR


#define BROADCAST_EXPEND 0
#define H_EXPEND 1
#define W_EXPEND 2
#define C_EXPEND 3
#define H_W_EXPEND 4
#define H_C_EXPEND 5
#define W_C_EXPEND 6
#define NO_EXPEND 7

 #ifdef OP_DEBUG 

static inline __attribute__((__overloadable__))void print_vec(vfloat32m8_t vec, int32_t inc)
{
    float_t observ[128];
    vse32_v_f32m8(observ, vec, inc);
    for (int i = 0; i < inc; i++){
    	if(observ[i] > -1.0 && observ[i] < 0.0)
    		debug_op("-");
    	debug_op("%d",(int32_t)observ[i]);
    	debug_op(".");
    	float_t temp = fabs(observ[i]-(int32_t)observ[i]);
    	for(int j = 0; j < 7; j++){
    		temp = temp*10;
    	   	debug_op("%u",(int32_t)temp);
    		temp = temp - (int32_t)temp;
    	}
    	debug_op("	");
   }
//    debug_op("\n");
}
static inline __attribute__((__overloadable__))void print_vec(vfloat32m1_t vec, int32_t inc)
{
    float_t observ[16];
    vse32_v_f32m1(observ, vec, inc);
	for (int i = 0; i < inc; i++){
	   	if(observ[i] > -1.0 && observ[i] < 0.0)
    		debug_op("-");
		debug_op("%d",(int32_t)observ[i]);
		debug_op(".");
		float_t temp = fabs(observ[i]-(int32_t)observ[i]);
		for(int j = 0; j < 7; j++){
			temp = temp*10;
			debug_op("%u",(int32_t)temp);
			temp = temp - (int32_t)temp;
		}
		debug_op("	");
	}
 //    debug_op("\n");
}

static inline __attribute__((__overloadable__))void print_vec(vint8m2_t vec, int32_t inc)
{
    int8_t observ[128];
    vse8_v_i8m2(observ, vec, inc);
    for (int i = 0; i < inc; i++)
        debug_op("%d	", observ[i]);
//    debug_op("\n");
}
 #else 
//static inline __attribute__((__overloadable__))void print_vec(vfloat32m8_t vec, int32_t inc){}
//static inline __attribute__((__overloadable__))void print_vec(vfloat32m1_t vec, int32_t inc){}
//static inline __attribute__((__overloadable__))void print_vec(vint8m2_t vec, int32_t inc){}
#endif
int32_t executeSoftmax(	struct cpu_param cpu_parameters);
//int32_t executePowerV2(struct cpu_param cpu_parameters);
int32_t executeReshape(struct cpu_param cpu_parameters);
int32_t executeUpsampe_nearest(struct cpu_param cpu_parameters);
int32_t executeConcat(struct cpu_param cpu_parameters);
int32_t executeSlice(struct cpu_param cpu_parameters);
int32_t executeAddV2_Sub_Mult_Div_Power(struct cpu_param cpu_parameters);
int32_t executeReduce(struct cpu_param cpu_parameters);
/* int32_t executeAddV2(struct cpu_param cpu_parameters); */
/* int32_t executeMultiply(struct cpu_param cpu_parameters); */
int32_t executeExpf_Tanh(struct cpu_param cpu_parameters);
int32_t executeSigmoid(struct cpu_param cpu_parameters);
int32_t executeResize(struct cpu_param cpu_parameters);
int32_t executeCast(struct cpu_param cpu_parameters);
int32_t executeDense(struct cpu_param cpu_parameters);
int32_t executeOne_hot(struct cpu_param cpu_parameters);
int32_t executeTake(struct cpu_param cpu_parameters);
int32_t executeTranspose(struct cpu_param cpu_parameters);
int32_t executeSqueeze(struct cpu_param cpu_parameters);
int32_t executeExpend_dims(struct cpu_param cpu_parameters);
int32_t executeSplit(struct cpu_param cpu_parameters);
int32_t executeBatch_matmul(struct cpu_param cpu_parameters);
int32_t executePool2D(struct cpu_param cpu_parameters);

#endif
