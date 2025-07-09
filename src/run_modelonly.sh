#!/bin/bash

VENAME=utvenvitemseg

echo "Activate the virtualenv"
source $VENAME/bin/activate



rm -rf segout01
python3 -m itemseg --input_type raw --input ../testdata/homedepot_raw.txt --method crf
diff segout01 ../testdata/out_homedepot_raw
# if $? -ne 0; then
#     echo "Error: raw output different from prestored result."
#     exit 1
# fi

