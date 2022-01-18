#!/bin/bash 

output='../example/data/ner_predict'

if [ ! -d $output ]; then 
    mkdir ${output}
fi

cp ../example/configs/predict.json.example ${output}/hf_argument.json

CUDA_VISIBLE_DEVICES=1 python predict.py {output}/hf_argument.json
