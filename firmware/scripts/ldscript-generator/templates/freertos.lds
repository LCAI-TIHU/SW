{% extends "base.lds" %}

{% block description %}
/* Privileged mode Linker Script
 *
 * This linker script is based on metal.default.lds. It introduce specific
 * section to isolate (acessible only from machine mode) and others that can be
 * used in every execution mode. This linker script it tailored for FreeRTOS
 * applications.
 */
{% endblock %}

{#
 # Set layout options
 #}
{% set ramrodata = False %}
{% set privilege_en = True %}
{% set eccscrub_en = True %}

{% block privileged_function_section %}
	.privileged_functions : ALIGN (32) {
        __privileged_functions_start__ = .;
        KEEP(*(privileged_functions))
        . = ALIGN(32);
		__privileged_functions_end__ = .;
	} >{{ rom.vma }}
{% endblock %} 

{% block privileged_data_section %}
    .privileged_data (NOLOAD) : ALIGN(32) {
        __privileged_data_start__ = .;
		*(privileged_data)
		/* Non kernel data is kept out of the first _Privileged_Data_Region_Size
		bytes of SRAM. */
        . = ALIGN(32);
		__privileged_data_end__ = .;
	} >{{ ram.vma }}
{% endblock %} 
