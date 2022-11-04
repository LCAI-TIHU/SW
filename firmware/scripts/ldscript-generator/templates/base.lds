/* Copyright (c) 2020 SiFive Inc. */
/* SPDX-License-Identifier: Apache-2.0 */
{# Checks to make sure that required template values are provided #}
{% if rom.vma is undefined %}
    {{ missingvalue("Cannot render linker script, romable regions don't specify lma/vma!") }}
{% endif %}
{% if itim.lma is undefined or itim.vma is undefined %}
    {{ missingvalue("Cannot render linker script, itimable regions don't specify lma/vma!") }}
{% endif %}
{% if lim.lma is undefined or lim.vma is undefined %}
    {{ missingvalue("Cannot render linker script, limable regions don't specify lma/vma!") }}
{% endif %}
{% if ram.lma is undefined or ram.vma is undefined %}
    {{ missingvalue("Cannot render linker script, ramable regions don't specify lma/vma!") }}
{% endif %}
{% if num_harts is undefined %}
    {{ missingvalue("Cannot render linker script, num_harts is not specified!") }}
{% endif %}
OUTPUT_ARCH("riscv")

{% block description %}{% endblock %}

ENTRY(_enter)

MEMORY
{
{% for memory in memories %}
    {{ memory.name }} ({{ memory.attributes }}) : ORIGIN = {{ memory.base_hex }}, LENGTH = {{ memory.length_hex }}
{% else %}
    {{ missingvalue("No memories are defined, cannot render linker script!") }}
{% endfor %}
}

PHDRS
{
    rom PT_LOAD;
    ram_init PT_LOAD;
    tls PT_TLS;
    ram PT_LOAD;
    itim_init PT_LOAD;
    text PT_LOAD;
    lim_init PT_LOAD;
}

SECTIONS
{
    /* Each hart is allocated its own stack of size __stack_size. This value
     * can be overriden at build-time by adding the following to CFLAGS:
     *
     *     -Xlinker --defsym=__stack_size=0xf00
     *
     * where 0xf00 can be replaced with a multiple of 16 of your choice.
     *
     * __stack_size is PROVIDE-ed as a symbol so that initialization code
     * initializes the stack pointers for each hart at the right offset from
     * the _sp symbol.
     */
    __stack_size = DEFINED(__stack_size) ? __stack_size : {{ default_stack_size|default("0x400") }};
    PROVIDE(__stack_size = __stack_size);

    /* The size of the heap can be overriden at build-time by adding the
     * following to CFLAGS:
     *
     *     -Xlinker --defsym=__heap_size=0xf00
     *
     * where 0xf00 can be replaced with the value of your choice.
     *
     * Altertatively, the heap can be grown to fill the entire remaining region
     * of RAM by adding the following to CFLAGS:
     *
     *     -Xlinker --defsym=__heap_max=1
     *
     * Note that depending on the memory layout, the bitness (32/64bit) of the
     * target, and the code model in use, this might cause a relocation error.
     */
    __heap_size = DEFINED(__heap_size) ? __heap_size : {{ default_heap_size|default("0x800") }};

    /* The boot hart sets which hart runs the pre-main initialization routines,
     * including copying .data into RAM, zeroing the BSS region, running
     * constructors, etc. After initialization, the boot hart is also the only
     * hart which runs application code unless the application overrides the
     * secondary_main() function to start execution on secondary harts.
     */
    PROVIDE(__metal_boot_hart = {{ boot_hart|default(0) }});

    /* The chicken bit is used by pre-main initialization to enable/disable
     * certain core features */
    PROVIDE(__metal_chicken_bit = {{ chicken_bit|default(0) }});

{% if eccscrub_en %}
    /* The memory_ecc_scrub bit is used by _entry code to enable/disable
     * memories scrubbing to zero  */
    PROVIDE(__metal_eccscrub_bit = {{ eccscrub_bit|default(0) }});

    /* The RAM memories map for ECC scrubbing */
{% for memory in ram_memories %}
    {% if '0x10000' in memory["length_hex"]: %}
    /* Default zero-scrub to at most 64KB, for limiting RTL simulation run time. */
    /* User is recommended to enable the full size for manual RTL simulation run! */
    {% endif %}
    PROVIDE( metal_{{ memory.name }}_memory_start = {{ memory.base_hex }} );
    PROVIDE( metal_{{ memory.name }}_memory_end = {{ memory.base_hex }} + {{ memory.length_hex }} );
{% endfor %}
{% else %}
    PROVIDE(__metal_eccscrub_bit = 0);
{% endif %}

    /* ROM SECTION
     *
     * The following sections contain data which lives in read-only memory, if
     * such memory is present in the design, for the entire duration of program
     * execution.
     */

    .init : {
        /* The _enter symbol is placed in the .text.metal.init.enter section
         * and must be placed at the beginning of the program */
        KEEP (*(.text.metal.init.enter))
        KEEP (*(.text.metal.init.*))
        KEEP (*(SORT_NONE(.init)))
        KEEP (*(.text.libgloss.start))
    } >{{ rom.vma }} :rom

    .fini : {
        KEEP (*(SORT_NONE(.fini)))
    } >{{ rom.vma }} :rom

    .preinit_array : ALIGN(8) {
        PROVIDE_HIDDEN (__preinit_array_start = .);
        KEEP (*(.preinit_array))
        PROVIDE_HIDDEN (__preinit_array_end = .);
    } >{{ rom.vma }} :rom

    .init_array : ALIGN(8) {
        PROVIDE_HIDDEN (__init_array_start = .);
        KEEP (*(SORT_BY_INIT_PRIORITY(.init_array.*) SORT_BY_INIT_PRIORITY(.ctors.*)))
        KEEP (*(.init_array EXCLUDE_FILE (*crtbegin.o *crtbegin?.o *crtend.o *crtend?.o ) .ctors))
        PROVIDE_HIDDEN (__init_array_end = .);
        PROVIDE_HIDDEN ( metal_constructors_start = .);
        KEEP (*(SORT_BY_INIT_PRIORITY(.metal.init_array.*)));
        KEEP (*(.metal.init_array));
        PROVIDE_HIDDEN ( metal_constructors_end = .);
    } >{{ rom.vma }} :rom

    .fini_array : ALIGN(8) {
        PROVIDE_HIDDEN (__fini_array_start = .);
        KEEP (*(SORT_BY_INIT_PRIORITY(.fini_array.*) SORT_BY_INIT_PRIORITY(.dtors.*)))
        KEEP (*(.fini_array EXCLUDE_FILE (*crtbegin.o *crtbegin?.o *crtend.o *crtend?.o ) .dtors))
        PROVIDE_HIDDEN (__fini_array_end = .);
        PROVIDE_HIDDEN ( metal_destructors_start = .);
        KEEP (*(SORT_BY_INIT_PRIORITY(.metal.fini_array.*)));
        KEEP (*(.metal.fini_array));
        PROVIDE_HIDDEN ( metal_destructors_end = .);
    } >{{ rom.vma }} :rom

{% block privileged_function_section %}{% endblock %} 

    .ctors : {
        {% if privilege_en %}
        . = ALIGN(32);
        __unprivileged_section_start__ = .;
        {% endif %}
        KEEP (*crtbegin.o(.ctors))
        KEEP (*crtbegin?.o(.ctors))
        KEEP (*(EXCLUDE_FILE (*crtend.o *crtend?.o ) .ctors))
        KEEP (*(SORT(.ctors.*)))
        KEEP (*(.ctors))
        KEEP (*(.metal.ctors .metal.ctors.*))
    } >{{ rom.vma }} :rom

    .dtors : {
        KEEP (*crtbegin.o(.dtors))
        KEEP (*crtbegin?.o(.dtors))
        KEEP (*(EXCLUDE_FILE (*crtend.o *crtend?.o ) .dtors))
        KEEP (*(SORT(.dtors.*)))
        KEEP (*(.dtors))
        KEEP (*(.metal.dtors .metal.dtors.*))
        {% if ramrodata and text_in_itim and privilege_en %}
        __unprivileged_section_end__ = .;
        {% endif %}
    } >{{ rom.vma }} : rom

{% if not ramrodata %}
    .rodata : {
        *(.rdata)
        *(.rodata .rodata.*)
        *(.gnu.linkonce.r.*)
        . = ALIGN(8);
        *(.srodata.cst16)
        *(.srodata.cst8)
        *(.srodata.cst4)
        *(.srodata.cst2)
        *(.srodata .srodata.*)
    } >{{ rom.vma }} :rom
{% endif %}

    /* ITIM SECTION
     *
     * The following sections contain data which is copied from read-only
     * memory into an instruction tightly-integrated memory (ITIM), if one
     * is present in the design, during pre-main program initialization.
     *
     * Generally, the data copied into the ITIM should be performance-critical
     * functions which benefit from low instruction-fetch latency.
     */

    .itim : ALIGN(8) {
        *(.itim .itim.*)
{% block force_itim %}
{% endblock %}
    } >{{ itim.vma }} AT>{{ itim.lma }} :itim_init

    PROVIDE( metal_segment_itim_source_start = LOADADDR(.itim) );
    PROVIDE( metal_segment_itim_target_start = ADDR(.itim) );
    PROVIDE( metal_segment_itim_target_end = ADDR(.itim) + SIZEOF(.itim) );

    /* LIM SECTION
     *
     * The following sections contain data which is copied from read-only
     * memory into a loosely integrated memory (LIM), which is shared with L2
     * cache, during pre-main program initialization.
     *
     * Generally, the data copied into the LIM should be performance-critical
     * functions which benefit from low instruction-fetch latency.
     */

    .lim : ALIGN(8) {
        *(.lim .lim.*)
{% block force_lim %}
{% endblock %}
    } >{{ lim.vma }} AT>{{ lim.lma }} :lim_init

    PROVIDE( metal_segment_lim_source_start = LOADADDR(.lim) );
    PROVIDE( metal_segment_lim_target_start = ADDR(.lim) );
    PROVIDE( metal_segment_lim_target_end = ADDR(.lim) + SIZEOF(.lim) );

    /* TEXT SECTION
     *
     * The following section contains the code of the program, excluding
     * everything that's been allocated into the ITIM/LIM already
     */

    .text : {
        *(.text.unlikely .text.unlikely.*)
        *(.text.startup .text.startup.*)
        *(.text .text.*)
        *(.gnu.linkonce.t.*)
        {% if privilege_en %}
        *(freertos_system_calls)
        . = ALIGN(32);
        __unprivileged_section_end__ = .;
        {% endif %}
    } >{{ rom.vma }} :text

    /* RAM SECTION
     *
     * The following sections contain data which is copied from read-only
     * memory into a read-write-capable memory such as data tightly-integrated
     * memory (DTIM) or another main memory, as well as the BSS, stack, and
     * heap.
     *
     * You might notice that .data, .tdata, .tbss, .tbss_space, and .bss all
     * have an apparently unnecessary ALIGN at their top. This is because
     * the implementation of _start in Freedom Metal libgloss depends on the
     * ADDR and LOADADDR being 8-byte aligned.
     */

    .data : ALIGN(8) {
        {% if privilege_en %}
        . = ALIGN(32);
        __unprivileged_data_section_start__ = .;
        {% endif %}
        *(.data .data.*)
        *(.gnu.linkonce.d.*)
        . = ALIGN(8);
        PROVIDE( __global_pointer$ = . + 0x800 );
        *(.sdata .sdata.* .sdata2.*)
        *(.gnu.linkonce.s.*)
{% if ramrodata %}
        /* Read-only data is placed in RAM to improve performance, since
         * read-only memory generally has higher latency than RAM */
        . = ALIGN(8);
        *(.srodata.cst16)
        *(.srodata.cst8)
        *(.srodata.cst4)
        *(.srodata.cst2)
        *(.srodata .srodata.*)
        . = ALIGN(8);
        *(.rdata)
        *(.rodata .rodata.*)
        *(.gnu.linkonce.r.*)
{% endif %}
    } >{{ ram.vma }} AT>{{ ram.lma }} :ram_init

    .tdata : ALIGN(8) {
        PROVIDE( __tls_base = . );
	*(.tdata .tdata.* .gnu.linkonce.td.*)
    } >{{ ram.vma }} AT>{{ ram.lma }} :tls :ram_init

    PROVIDE( __tdata_source = LOADADDR(.tdata) );
    PROVIDE( __tdata_size = SIZEOF(.tdata) );

    PROVIDE( metal_segment_data_source_start = LOADADDR(.data) );
    PROVIDE( metal_segment_data_target_start = ADDR(.data) );
    PROVIDE( metal_segment_data_target_end = ADDR(.tdata) + SIZEOF(.tdata) );

    .tbss : ALIGN(8) {
	*(.tbss .tbss.* .gnu.linkonce.tb.*)
	*(.tcommon .tcommon.*)
	PROVIDE( __tls_end = . );
    } >{{ ram.vma }} AT>{{ ram.vma }} :tls :ram
    PROVIDE( __tbss_size = SIZEOF(.tbss) );
    PROVIDE( __tls_size = __tls_end - __tls_base );

    .tbss_space : ALIGN(8) {
	. = . + __tbss_size;
    } >{{ ram.vma }} :ram

    .bss (NOLOAD): ALIGN(8) {
        *(.sbss*)
        *(.gnu.linkonce.sb.*)
        *(.bss .bss.*)
        *(.gnu.linkonce.b.*)
        *(COMMON)
{% if privilege_en %}
        . = ALIGN(32);
        __unprivileged_data_section_end__ = .;
{% endif %}
    } >{{ ram.vma }} :ram

    PROVIDE( metal_segment_bss_source_start = LOADADDR(.tbss) );
    PROVIDE( metal_segment_bss_target_start = ADDR(.tbss) );
    PROVIDE( metal_segment_bss_target_end = ADDR(.bss) + SIZEOF(.bss) );

{% block privileged_data_section %}{% endblock %} 

    .stack (NOLOAD) : ALIGN(16) {
        PROVIDE(metal_segment_stack_begin = .);
        . += __stack_size; /* Hart 0 */
        PROVIDE( _sp = . );
{% for hart in range(num_harts - 1) %}
        . += __stack_size; /* Hart {{ hart + 1 }} */
{% endfor %}
        PROVIDE(metal_segment_stack_end = .);
    } >{{ ram.vma }} :ram

    .heap (NOLOAD) : ALIGN(8) {
        PROVIDE( __end = . );
        PROVIDE( __heap_start = . );
        PROVIDE( metal_segment_heap_target_start = . );
        /* If __heap_max is defined, grow the heap to use the rest of RAM,
         * otherwise set the heap size to __heap_size */
        . = DEFINED(__heap_max) ? MIN( LENGTH(ram) - ( . - ORIGIN(ram)) , 0x10000000) : __heap_size;
        PROVIDE( metal_segment_heap_target_end = . );
        PROVIDE( _heap_end = . );
        PROVIDE( __heap_end = . );
    } >{{ ram.vma }} :ram

    /* C++ exception handling information is
     * not useful with our current runtime environment,
     * and it consumes flash space. Discard it until
     * we have something that can use it
     */
    /DISCARD/ : {
	*(.eh_frame .eh_frame.*)
    }
}
