#!/bin/bash
source ../env.sh
python3 from_tensorflow_quantize_bert.py 2>&1 | tee bert.log
