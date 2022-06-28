#!/bin/bash

config='preprocess_train.json'

# remove this line for your own task
cp ../example/configs/preprocess_train.json.example $config

python -u main.py --args_path $config > main.log 2>&1 &
