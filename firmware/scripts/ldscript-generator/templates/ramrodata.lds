{% extends "base.lds" %}

{% block description %}
/* RAM Read-Only Data Linker Script
 *
 * This linker script places application code and read-only data into writable
 * memories in an attempt to improve performance, since writable memories
 * are generally lower-latency. This linker script may cause your application
 * to overflow RAM, since it dramatically increases the quantity of data vying
 * for space there.
 */
{% endblock %}

{#
 # Set layout options
 #}
{% set ramrodata = True %}
{% set eccscrub_en = True %}

{% block force_itim %}
{% if text_in_itim %}
        /* The following takes advantage of -ffunction sections to link benchmark
         * code into the ITIM when the ITIM is big enough to take advantage of it.
         */
        *(.text.startup.main)

        /* Dhrystone */
        *(.text.Func_1)
        *(.text.Func_2)
        *(.text.Func_3)
        *(.text.malloc)
        *(.text.memchr)
        *(.text.memcpy)
        *(.text.memmove)
        *(.text.Proc_1)
        *(.text.Proc_2)
        *(.text.Proc_3)
        *(.text.Proc_4)
        *(.text.Proc_5)
        *(.text.Proc_6)
        *(.text.Proc_7)
        *(.text.Proc_8)
        *(.text.time)
        *(.text.strcmp)
        *strcmp.o(.text)

        /* Coremark */
        *(.text.barebones_clock)
        *(.text.calc_func)
        *(.text.check_data_types)
        *(.text.clock)
        *(.text.cmp_complex)
        *(.text.cmp_idx)
        *(.text.copy_info)
        *(.text.core_bench_list)
        *(.text.core_bench_matrix)
        *(.text.core_bench_state)
        *(.text.core_init_matrix)
        *(.text.core_init_state)
        *(.text.core_list_find)
        *(.text.core_list_init)
        *(.text.core_list_insert_new)
        *(.text.core_list_mergesort)
        *(.text.core_list_remove)
        *(.text.core_list_reverse)
        *(.text.core_list_undo_remove)
        *(.text.core_state_transition)
        *(.text.crc16)
        *(.text.crcu16)
        *(.text.crcu32)
        *(.text.crcu8)
        *(.text.get_seed_32)
        *(.text.get_time)
        *(.text.iterate)
        *(.text.matrix_add_const)
        *(.text.matrix_mul_const)
        *(.text.matrix_mul_matrix)
        *(.text.matrix_mul_matrix_bitextract)
        *(.text.matrix_mul_vect)
        *(.text.matrix_sum)
        *(.text.matrix_test)
        *(.text.memchr)
        *(.text.memcpy)
        *(.text.memmove)
        *(.text.portable_fini)
        *(.text.portable_init)
        *(.text.start_time)
        *(.text.stop_time)
        *(.text.time_in_secs)
        *(.text.unlikely.core_bench_list)
        *(.text.unlikely.core_list_init)
        *(.text.unlikely.core_list_mergesort)
{% endif %}
{% endblock %}
