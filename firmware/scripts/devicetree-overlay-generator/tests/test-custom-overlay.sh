#!/usr/bin/env bash

set -euo pipefail

OUTPUT_PATH="build/devicetree-overlay-generator/customOverlay.dts"

wake --init .

wake -v "writeDevicetreeCustomOverlay \"${OUTPUT_PATH}\" \
                                      (makeDevicetreeCustomOverlay (\"core.dts\", Nil) \
                                                                   (makeDevicetreeChosenNode (makeDevicetreeChosenMemoryEntry \"/soc/bootrom@20000000\" 0 0) \
                                                                                             (makeDevicetreeChosenMemoryRam \"/soc/sram@80000000\" 0 0) \
                                                                                             None))"

>&2 echo "$0: Checking for ${OUTPUT_PATH}"
if [ ! -f ${OUTPUT_PATH} ] ; then
        >&2 echo "$0: ERROR Failed to produce ${OUTPUT_PATH}"
        exit 1
fi

>&2 echo "$0: Checking for non-empty ${OUTPUT_PATH}"
if [ `grep -c '/include/ "core.dts"' ${OUTPUT_PATH}` -ne 1 ] ; then
        >&2 echo "$0: ERROR ${OUTPUT_PATH} has bad contents"
        exit 2
fi
