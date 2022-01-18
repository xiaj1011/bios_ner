#!/bin/bash 

output='../example/data/ner_train'

if [ ! -d $output ]; then 
    mkdir ${output}
fi

cp ../example/configs/train.json.example ${output}/hf_argument.json

CUDA_VISIBLE_DEVICES=1,2 python run_ner.py ${output}/hf_argument.json

