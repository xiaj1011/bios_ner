#!/bin/bash

output='../example/data/ner_server'

if [ ! -d $output ]; then 
    mkdir ${output}
fi

cp ../example/configs/server.json.example ${output}/hf_argument.json

CUDA_VISIBLE_DEVICES=1 python server.py ${output}/hf_argument.json 
