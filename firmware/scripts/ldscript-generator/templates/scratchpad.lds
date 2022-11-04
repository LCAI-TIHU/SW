{% extends "base.lds" %}

{% block description %}
/* Scratchpad Linker Script
 *
 * This linker script is for executing in "scratchpad" mode, where all
 * application code and data is placed in writable memory.
 */
{% endblock %}

{#
 # Set layout options
 #}
{% set ramrodata = False %}
{% set eccscrub_en = False %}
