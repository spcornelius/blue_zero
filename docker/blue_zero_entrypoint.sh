#!/bin/bash
git clone https://${GH_TOKEN}@github.com/spcornelius/blue_zero.git
cd blue_zero
conda run -n blue_zero python setup.py install
exec "$@"
