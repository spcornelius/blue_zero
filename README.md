# BLUE Zero

All of the stuff you will use day-to-day is in `/scripts`.

## Quick Start
0. Go into the scripts directory
```commandline
cd scripts
```

**1. Generate some training and validation boards of size 10, for mode 0**

    ```commandline
    python generate.py --num-boards 10000 -n 10 -p 0.65 -o  train_data.npy --mode 0
    python generate.py --num-boards 100 -n 10 -p 0.65 -o  validation_data.npy --mode 0
    ```

**2. Train a network on the CPU using the example hyper parameters specified
in `examples/10/config.yml`**

```commandline
    python train.py -c ../examples/10/config.yml -v validation_set.npy -t train_set.npy -o model.pt --device 'cpu' --mode 0
```

To use different parameters, just modify the YAML file (but make a copy; the examples directory is
version-controlled!)

**3. View the model's performance on an example game**

    ```commandline
    python view.py -f model.pt -n 10 -p 0.65 --mode 0 --pause 1
    ```
   
**4. Play around with different modes**

All of the scripts above accept a `--mode` argument at the command line. Currently,
0 and 3 are supported. For mode 3, there is an optional `--direction` argument
(default 'horizontal') specifying the direction of current flow for the game.