#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(dirname $0)
SPIKE_DTS_DIR=${SCRIPT_DIR}/spike

OUTPUT_PATH="build/cmsis-svd-generator/design.svd"

wake --init .

wake -v -x "runSVDGenerator (makeSVDGeneratorOptions (source \"${SPIKE_DTS_DIR}/design.dts\") (sources \"${SPIKE_DTS_DIR}\" \`core.dts\`) \"${OUTPUT_PATH}\")"

>&2 echo "$0: Checking for ${OUTPUT_PATH}"
if [ ! -f ${OUTPUT_PATH} ] ; then
        >&2 echo "$0: ERROR Failed to produce ${OUTPUT_PATH}"
        exit 1
fi

>&2 echo "$0: Checking for non-empty ${OUTPUT_PATH}"
if [ `grep -c 'CMSIS-SVD.xsd' ${OUTPUT_PATH}` -ne 1 ] ; then
        >&2 echo "$0: ERROR ${OUTPUT_PATH} has bad contents"
        exit 2
fi
