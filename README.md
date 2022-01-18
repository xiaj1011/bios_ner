# BIOS NER
This project is used for **named entity recognition**(NER) in the **medical** field, and it helps us build a large-scale medical knowledge graph BIOS (https://bios.idea.edu.cn/).



## NER Pipeline
A high-quality training set is the basis for a reliable supervised learning NER model, for which some additional work is required to implement training set annotation. The complete pipeline is shown in the figure below, and the details of some dependencies could be found in other projects under this group.
![flowchart of ner pipeline](./doc/ner_pipeline_20220116.png)

## Install
It is strongly recommended to train and predict on the GPU, so make sure you have the GPU and the correct CUDA installed on your device before working.
```commandline
sh env_setup.sh
```

## Modules
The following diagrams help you understand each module in the project, details can be found in each directory.
![ner modules](./doc/ner_modules.png)


## Citation
TODO
