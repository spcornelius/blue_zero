#/usr/bin/bash

python3 generate.py --num-boards $N_TRAIN_BOARDS -n $BOARD_SIZE -p $BLUE_ZERO_P -o trainboards --mode $BLUE_ZERO_MODE
python3 generate.py --num-boards $N_VAL_BOARDS -n $BOARD_SIZE -p $BLUE_ZERO_P -o valboards --mode $BLUE_ZERO_MODE
python3 train.py -c ../examples/10/config.yml -t trainboards.npy -v valboards.npy -o m4model.pt --mode 4 --device cuda -K $BLUE_ZERO_MODE_4_K
