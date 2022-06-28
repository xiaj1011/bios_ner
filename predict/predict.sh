#!/bin/bash 

output='../example/data/ner_predict'

if [ ! -d $output ]; then 
    mkdir ${output}
fi

# make sure you have trained model if you use this example script. Refer to train.sh from ../train
cp ../example/configs/predict.json.example ${output}/hf_argument.json
cp ../example/data/test/test.txt ${output}/test.txt

CUDA_VISIBLE_DEVICES=1 python -u predict.py ${output}/hf_argument.json > predict.log 2>&1 &
