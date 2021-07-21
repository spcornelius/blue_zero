# BLUE Zero

All of the stuff you will use day-to-day is in `/scripts`.

## Quick Start
**1. Go into the scripts directory**
```commandline
cd scripts
```

**2. Generate some training and validation boards of size 20, for mode 0**

```commandline
python generate.py --num-boards 10000 -n 20 -p 0.65 -o  train_data.npz --mode 0
python generate.py --num-boards 100 -n 20 -p 0.65 -o  validation_data.npz --mode 0
```

**3. Train a network on the GPU using the example hyper parameters specified
in `examples/10/config.yml`**

```commandline
python train.py -c ../examples/10/config.yml -v validation_set.npz -t train_set.npz -o train_results.pkl --device cuda
```

To use different parameters, just modify the YAML file (but make a copy; the examples directory is
version-controlled!)

[comment]: <> (**4. View the model's performance on an example game**)

[comment]: <> (```commandline)

[comment]: <> (python view.py -f model.pt -n 10 -p 0.65 --mode 0 --pause 1)

[comment]: <> (```)
   
[comment]: <> (**5. Play around with different modes**)

[comment]: <> (All of the scripts above accept a `--mode` argument at the command line. Currently,)

[comment]: <> (0 and 3 are supported. For mode 3, there is an optional `--direction` argument)

[comment]: <> (&#40;default 'horizontal'&#41; specifying the direction of current flow for the game.)