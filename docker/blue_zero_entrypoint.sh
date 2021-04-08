#!/bin/bash
git clone https://${GH_TOKEN}@github.com/spcornelius/blue_zero.git:noodle_clean
cd blue_zero
conda run -n blue_zero python setup.py install
echo $BLUE_ZERO_MODE
echo $BLUE_ZERO_P
echo $N_TRAIN_BOARDS
echo $N_VAL_BOARDS
echo $BLUE_ZERO_MODE_4_K
echo $BOARD_SIZE
conda run -n blue_zero bash ./scripts/fullpipeline.sh
exec "$@"
-e GH_TOKEN=661740da01e0fb094c25f65507ede87f008cadcc -e BLUE_ZERO_MODE=4 -e BLUE_ZERO_P=0.65 -e N_TRAIN_BOARDS=1000 -e N_VAL_BOARDS=100 -e BLUE_ZERO_MODE_4_K=1.5 -e BOARD_SIZE=20