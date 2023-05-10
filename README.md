# BLUE Zero

Examples for how to train appear in `/scripts`.

## Quick Start
1. Go into the scripts directory
```commandline
cd scripts
```

2. Generate some training and validation boards of size 5, for the "network" mode:

```commandline
python generate.py --num-boards 10000 -n 5 -p 0.65 -o  train_data_n5.npz --mode 0
python generate.py --num-boards 100 -n 5 -p 0.65 -o  validation_data_n5.npz --mode 0
```

3. Train a network on the GPU using the example hyper parameters specified
in `examples/config.yml`

```commandline
python train.py -c ../examples/config.yml -v validation_data_n5.npz -t train_data_n5.npz -o train_results_n5.pkl.gz --device cuda
```

To use different parameters, just modify the YAML file.
