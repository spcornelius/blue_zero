#!/bin/bash
git clone --branch noodle_clean https://${GH_TOKEN}@github.com/spcornelius/blue_zero.git
cd blue_zero
conda run -n blue_zero python setup.py install
echo $BLUE_ZERO_MODE
echo $BLUE_ZERO_P
echo $N_TRAIN_BOARDS
echo $N_VAL_BOARDS
echo $BLUE_ZERO_MODE_4_K
echo $BOARD_SIZE
pushd scripts
conda run -n blue_zero bash fullpipeline.sh
popd
exec "$@"
