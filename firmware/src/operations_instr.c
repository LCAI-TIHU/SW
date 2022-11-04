/*
 * operations_instr.c
 *
 *  Created on: Dec 10, 2021
 *      Author: root
 */
/*
Based on:

		e ^ x = (1+m) * (2^n)
		x = log(1+m) + n * log(2)
		n = (int) (x * 1.0 / log(2))
		(1+m) = e ^ (x - n * log(2))
		(1+m) = Poly(x - n * log(2))

		where Poly(x) is the Minimax approximation of e ^ x over the
		range [-Log(2), Log(2)]

Test func : expf(x)
Test Range: 0 < x < 50
Peak Error:	~0.00024%
RMS  Error: ~0.00007%
*/

#include <stdint.h>
//#include <math.h>
#include <stdlib.h>
//#include "cpu_interface.h"
#include "cpu_callback.h"

#include "operations.h"
//#include <metal/drivers/sifive_pl2cache0.h>

#ifdef __MATH_VECTOR

#define X280_VLEN  512
#define ENUM_E32M8 X280_VLEN/32*8
#define ENUM_E32M1 X280_VLEN/32

extern struct cpu_device cpu_dev;

const float_t __expf_rng[2] = {
	1.442695041f,
	0.693147180f
};

const float_t __expf_lut[8] = {
	0.9999999916728642,		//p0
	0.04165989275009526, 	//p4
	0.5000006143673624, 	//p2
	0.0014122663401803872, 	//p6
	1.000000059694879, 		//p1
	0.008336936973260111, 	//p5
	0.16666570253074878, 	//p3
	0.00019578093328483123	//p7
};

vfloat32m8_t expf_vector(vfloat32m8_t x, uint32_t inc)
{

	//Range Reduction:
//	m = (int) (x * __expf_rng[0]);
//	x = x - ((float) m) * __expf_rng[1];
	vfloat32m8_t m_temp = vfmul_vf_f32m8 (x, __expf_rng[0], inc);
	vint32m8_t m_int = vfcvt_x_f_v_i32m8 (m_temp, inc);
	vfloat32m8_t m_float = vfcvt_f_x_v_f32m8 (m_int, inc);
//	debug_op("\nm_float:\n");
//	print_vec(m_float, inc);
	vfloat32m8_t x_temp = vfmul_vf_f32m8 (m_float, __expf_rng[1], inc);
//	debug_op("\n x_temp :\n");
//	print_vec(x_temp, inc);
	x = vfsub_vv_f32m8 (x, x_temp, inc);
//	debug_op("\n x :\n");
//	print_vec(x, inc);
	//	a = (__expf_lut[4] * x) + (__expf_lut[0]);
	vfloat32m8_t a_temp = vfmul_vf_f32m8 (x, __expf_lut[4], inc);
	vfloat32m8_t a = vfadd_vf_f32m8(a_temp, __expf_lut[0], inc);
//	debug_op("\n a: \n");
//	print_vec(a, inc);
//	b = (__expf_lut[6] * x) + (__expf_lut[2]);
	vfloat32m8_t b_temp = vfmul_vf_f32m8 (x, __expf_lut[6], inc);
	vfloat32m8_t b = vfadd_vf_f32m8(b_temp, __expf_lut[2], inc);
//	debug_op("\n b: \n");
//	print_vec(b, inc);
//	c = (__expf_lut[5] * x) + (__expf_lut[1]);
	vfloat32m8_t c_temp = vfmul_vf_f32m8 (x, __expf_lut[5], inc);
	vfloat32m8_t c = vfadd_vf_f32m8(c_temp, __expf_lut[1], inc);
//	debug_op("\n c: \n");
//	print_vec(c, inc);
//	d = (__expf_lut[7] * x) + (__expf_lut[3]);
	vfloat32m8_t d_temp = vfmul_vf_f32m8 (x, __expf_lut[7], inc);
	vfloat32m8_t d = vfadd_vf_f32m8(d_temp, __expf_lut[3], inc);
//	debug_op("\n d: \n");
//	print_vec(d, inc);
//	xx = x * x;
	vfloat32m8_t xx = vfmul_vv_f32m8(x, x, inc);
//	a = a + b * xx;
	vfloat32m8_t b_xx = vfmul_vv_f32m8(b, xx, inc);
	a = vfadd_vv_f32m8(a, b_xx, inc);
//	c = c + d * xx;
	vfloat32m8_t d_xx = vfmul_vv_f32m8(d, xx, inc);
	c = vfadd_vv_f32m8(c, d_xx, inc);
//	xx = xx* xx;
	xx = vfmul_vv_f32m8(xx, xx, inc);
//	r.f = a + c * xx;
	vfloat32m8_t c_xx = vfmul_vv_f32m8(c, xx, inc);
	vfloat32m8_t r_f = vfadd_vv_f32m8(a, c_xx, inc);
//	debug_op("\n r_f: \n");
//	print_vec(r_f, inc);
	//multiply by 2 ^ m
//	m = m << 23;
//	r.i = r.i + m;
	m_int = vsll_vx_i32m8 (m_int, 23, inc);
	__asm__ volatile("vadd.vv %0, %1, %2"
			: "=vr"(r_f)
			: "vr"(m_int), "vr"(r_f)
			:);
//	debug_op("\n r_f: \n");
//	print_vec(r_f, inc);
	return r_f;

/*
    float_t observ[128];
    float_t out[128];
    vse32_v_f32m8(observ, x, inc);
	float_t a, b, c, d, xx;
	int32_t m;

	union {
		float_t   f;
		int32_t 	i;
	} r;
	debug_op("\n");
    for (int32_t i = 0; i < inc; i++){
		//Range Reduction:
    	float_t x_i = observ[i];
		m = (int) (x_i * __expf_rng[0]);
		debug_op(" m:");
		print_float((float)m);
		x_i = x_i - ((float) m) * __expf_rng[1];
		debug_op(" x_i:");
		print_float(x_i);

		//Taylor Polynomial (Estrins)
		a = (__expf_lut[4] * x_i) + (__expf_lut[0]);
		debug_op(" a %d:", (int32_t)(a*1000000));
		print_float(a);
		debug_op(" b:");
		b = (__expf_lut[6] * x_i) + (__expf_lut[2]);
		print_float(b);
		debug_op(" c:");
		c = (__expf_lut[5] * x_i) + (__expf_lut[1]);
		print_float(c);
		debug_op(" d:");
		d = (__expf_lut[7] * x_i) + (__expf_lut[3]);
		print_float(d);
		xx = x_i * x_i;
		a = a + b * xx;
		c = c + d * xx;
		xx = xx* xx;
		r.f = a + c * xx;
		debug_op(" r_f:");
		print_float(r.f);
		//multiply by 2 ^ m
		m = m << 23;
		r.i = r.i + m;
		out[i] = r.f;
		debug_op(" r_f:");
		print_float(r.f);
    }

	return vle32_v_f32m8(out, inc);
*/
}

int32_t executeSoftmax(struct cpu_param cpu_parameters)
{
	debug_trace("Enter %s\n", __func__);

	int32_t err = 0;
    uint32_t ii = 0;
#if 0
	struct cpu_buffer_desc src = softmax_buffer_descs->src_data;
	struct cpu_buffer_desc dst = softmax_buffer_descs->dst_data;

    debug_trace("Processing softmax [axis=%u]\n", softmax_op_desc->axis);
    debug_trace("src format %u\n", src.format);
    debug_trace("\taddress[%u][%u] 0x%08x (%ux%ux%u) %uB\n", src.addressIndex, src.addressIndexOffset,
            addressList[src.addressIndex], src.width, src.height, src.channel, src.size);
    debug_trace("\tline_stride %uB surface_stride %uB\n", src.line_stride, src.surf_stride);
    debug_trace("\tinput scale factor: %f, output scale factor: %f\n", common_op_desc->input_scale_factor, common_op_desc->output_scale_factor);

    debug_trace("dst format %u\n", dst.format);
    debug_trace("\taddress[%u][%u] 0x%08x (%ux%ux%u) %uB\n", dst.addressIndex,  dst.addressIndexOffset,
            addressList[dst.addressIndex], dst.width, dst.height, dst.channel, dst.size);
    debug_trace("\tline_stride %uB surface_stride %uB\n", dst.line_stride, dst.surf_stride);
    if(src.format != dst.format)
    {
        debug_trace("Don't support CPU Scale operation with different src(%d) and dst(%d) formats\n",src.format, dst.format);
        err = 1;
        return err;
    }
#endif

    int32_t h, w, c;

    uint64_t output_addr_temp, input_addr_temp;
    
    uint64_t pDst = cpu_dev.task->address_list[cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.index];
    uint32_t dat_h_out = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.height;
    uint32_t dat_w_out = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.width;
    uint32_t dat_c_out = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.channel;
    int32_t line_stride_out = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.line_stride;
    int32_t surf_stride_out = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.surf_stride;
    uint32_t dat_type_out = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.dst_data.datatype;
    uint32_t dat_type_out_size = (dat_type_out == RINT8) ? 1 : 
                                (dat_type_out == RBFLOAT) ? 2 : 4;
    uint32_t size_out = dat_h_out * dat_w_out * dat_c_out;
    
    uint64_t pSrc = cpu_dev.task->address_list[cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.index];
    uint32_t dat_type_in = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.datatype;
    uint32_t dat_type_in_size = (dat_type_in == RINT8) ? 1 : 
                                (dat_type_in == RBFLOAT) ? 2 : 4;

    uint32_t dat_h_in = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.height;
    uint32_t dat_w_in = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.width;
    uint32_t dat_c_in = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.channel;
    int32_t line_stride_in = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.line_stride;
    int32_t surf_stride_in = cpu_parameters.cpu_operation_buffer.common_only_op_buffers.src_data.surf_stride;
    uint32_t size_in = dat_h_in * dat_w_in * dat_c_in;

		float *pHalfSrc = (float *)(malloc(dat_c_in * sizeof(float)));
		vint8m2_t 		src_int8m2;
		vfloat16m4_t 	src_f16m4;
		vfloat32m8_t    src_f32m8;
		vfloat32m1_t    max_f32m1;

		int8_t *pSrc_temp = pSrc;    //数据地址
		float *pHalfSrc_temp = pHalfSrc; //分配的地址
		uint32_t op_num, len = dat_c_in;


	//	vfmv_s_f_f32m1 (max_f32m1, (float_t)(-INFINITY), 16);  //将负无穷放到max_f32m1

		max_f32m1 = vfmv_v_f_f32m1((float_t)(-INFINITY),ENUM_E32M1);
		debug_op("\n max_f32m1:\n");
		print_vec(max_f32m1, ENUM_E32M1);

//		#pragma clang loop vectorize(enable)
		for(; (op_num = vsetvl_e32m8 (len))>0; len -= op_num)
		{

			src_f32m8 = vle32_v_f32m8 (pSrc_temp, op_num);  //将src放到src_f32m8中
			pSrc_temp = pSrc_temp + op_num;              // 地址加 64
			debug_op("\n%d src_float before scale:\n", op_num);
			print_vec(src_f32m8, op_num);
			vse32_v_f32m8 (pHalfSrc_temp, src_f32m8, op_num); //将数据存到分配的地址空间中
			max_f32m1 = vfredmax_vs_f32m8_f32m1(max_f32m1, src_f32m8, max_f32m1, op_num); //找最大值
			debug_op("\n %d src_float max:\n",op_num);
			print_vec(max_f32m1, op_num);
			pHalfSrc_temp = pHalfSrc_temp + op_num*4;
		}

		float_t maxval = vfmv_f_s_f32m1_f32 (max_f32m1); // 把最大值放到maxal中
		debug_op("\nMax value is :\n");
		print_float(maxval);
		
		vfloat32m1_t sumexp_vec = vfmv_v_f_f32m1 ((float_t)(0.0), ENUM_E32M1);
		debug_op("\n sumexp_vec:\n");
		print_vec(sumexp_vec, ENUM_E32M1);

		pHalfSrc_temp = pHalfSrc;
		pSrc_temp = pSrc; 
		len = dat_c_in;
//		#pragma clang loop vectorize(enable)
		for(; (op_num = vsetvl_e32m8 (len))>0; len -= op_num)
		{
			src_f32m8 = vle32_v_f32m8 (pSrc_temp, op_num);
			pSrc_temp = pSrc_temp + op_num; 
			debug_op("\n%d src_temp:\n", op_num);
			print_vec(src_f32m8, op_num);

			src_f32m8 = vfsub_vf_f32m8(src_f32m8, maxval, op_num);
			debug_op("\n%d src_float - maxval:\n", op_num);
			print_vec(src_f32m8, op_num);

			vfloat32m8_t sumexp_temp = expf_vector(src_f32m8, op_num);
			debug_op("\n%d exp of src_float:\n", op_num);
			print_vec(sumexp_temp, op_num);

			sumexp_vec = vfredsum_vs_f32m8_f32m1 (sumexp_vec, sumexp_temp, sumexp_vec, op_num);
			debug_op("\n %d sum of src_float_exp:\n",op_num);
			print_vec(sumexp_vec, op_num);

			pHalfSrc_temp = pHalfSrc_temp + op_num*4;
            
		}
		
		float sumexp = vfmv_f_s_f32m1_f32 (sumexp_vec);
		debug_op("\nSum exp is :\n");
		print_float(sumexp);

		pHalfSrc_temp = pHalfSrc;
		int8_t *pDst_temp = pDst;
		len = dat_c_in;
	//	float output_scale_factor = cpu_parameters.cpu_operation.common_only_op.common.output_scale_factor[0];
	//	debug_op("\n output_scale_factor:%u\n", output_scale_factor);

//		#pragma clang loop vectorize(enable)
		for(; (op_num = vsetvl_e32m8 (len))>0; len -= op_num)
		{
			vfloat32m8_t src_temp = vle32_v_f32m8 (pHalfSrc_temp, op_num);
				debug_op("\n%d 1src_temp:\n", op_num);
			print_vec(src_temp, op_num);

			pHalfSrc_temp = pHalfSrc_temp + op_num*4;
			src_temp = vfsub_vf_f32m8(src_temp, maxval, op_num);
				debug_op("\n%d 2src_temp:\n", op_num);
			print_vec(src_temp, op_num);

			vfloat32m8_t exp_temp = expf_vector(src_temp, op_num);
				debug_op("\n%d exp_temp:\n", op_num);
			print_vec(exp_temp, op_num);

			vfloat32m8_t dst_temp = vfdiv_vf_f32m8(exp_temp, sumexp, op_num);
				debug_op("\n%d exp_temp:\n", op_num);
			print_vec(dst_temp, op_num);

		//	dst_temp = vfdiv_vf_f32m8(dst_temp, output_scale_factor, op_num);
		//	debug_op("\n%d Softmax out in float:\n", op_num);
		//	print_vec(dst_temp, op_num);

			//dst_temp = vfdiv_vf_f32m8(dst_temp, (float)cpu_parameters.cpu_operation.common_only_op.common.output_scale_factor, op_num);

	
			vbool4_t mask0 = vmfgt_vf_f32m8_b4 (dst_temp, 127.0, op_num);

			dst_temp = vfmerge_vfm_f32m8 (mask0, dst_temp, 127.0, op_num);
						debug_op("\n%d mask0:\n", op_num);
			print_vec(dst_temp, op_num);

			vbool4_t mask1 = vmflt_vf_f32m8_b4 (dst_temp, -128.0, op_num);



			dst_temp = vfmerge_vfm_f32m8 (mask1, dst_temp, -128.0, op_num);
			debug_op("\n%d dst_temp:\n", op_num);
			print_vec(dst_temp, op_num);

		//	vint16m4_t dst_int16 = vfncvt_x_f_w_i16m4 (dst_temp, op_num);

		//	vint8m2_t dst_int8 = vncvt_x_x_w_i8m2 (dst_int16, op_num);
		//		debug_op("\n%d dst_int8:\n", op_num);
		//	print_vec(dst_int8, op_num);

		//	     vse8_v_i8m2 (pDst_temp, dst_int8, op_num);

 			vse32_v_f32m8 (pDst_temp, dst_temp, op_num);
			pDst_temp = pDst_temp + op_num;
		}
		debug_op("X280 OUTPUT ADDR is 0x%08x, DATAs are:\n", pDst);
		for(ii=0; ii<dat_c_out; ii++)
		{
		//	debug_trace("\t%d\n",pDst[ii]);
		}
		free(pHalfSrc);
	//}
    debug_trace("Exit %s\n", __func__);
    return err;
}


#endif
