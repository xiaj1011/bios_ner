#!/bin/bash 

output='../example/data/ner_train'
if [ ! -d $output ]; then 
    mkdir ${output}
fi

# download pretrained models firstly if they are not exist. Refer to download scripts from ../pretrain/
ln -s $HOME/models/pubmedbert_abs ../pretrain/pubmedbert_abs
cp ../example/configs/train.json.example ${output}/hf_argument.json

CUDA_VISIBLE_DEVICES=1,2 python run_ner.py ${output}/hf_argument.json

