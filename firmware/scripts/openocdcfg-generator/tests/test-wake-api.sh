#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(dirname $0)
HIFIVE1_DTS_DIR=${SCRIPT_DIR}/hifive1

OUTPUT_PATH="build/openocdcfg-generator/openocd.cfg"

wake --init .

wake -v "runOpenOCDConfigGenerator (makeOpenOCDConfigGeneratorOptions (source \"${HIFIVE1_DTS_DIR}/design.dts\") (sources \"${HIFIVE1_DTS_DIR}\" \`core.dts\`) \"hifive\" \"${OUTPUT_PATH}\")"

>&2 echo "$0: Checking for ${OUTPUT_PATH}"
if [ ! -f ${OUTPUT_PATH} ] ; then
        >&2 echo "$0: ERROR Failed to produce ${OUTPUT_PATH}"
        exit 1
fi

>&2 echo "$0: Checking for non-empty ${OUTPUT_PATH}"
if [ `grep -c 'adapter_khz' ${OUTPUT_PATH}` -ne 1 ] ; then
        >&2 echo "$0: ERROR ${OUTPUT_PATH} has bad contents"
        exit 2
fi
