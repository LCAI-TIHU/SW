{% extends "base.lds" %}

{% block description %}
/* Default Linker Script
 *
 * This is the default linker script for all Freedom Metal applications.
 */
{% endblock %}

{#
 # Set layout options
 #}
{% set ramrodata = False %}
{% set eccscrub_en = True %}
