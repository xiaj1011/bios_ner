#!/bin/bash

dst=$HOME/models/pubmedbert_abs
if [ ! -d $dst ];
then
    echo "create directory $dst"
    mkdir -p $dst
else
    echo "$dst already exists, skip download"
    exit 0
fi

git lfs install
git clone https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract $dst

