#!/usr/bin/env python
"""
NER的预测代码
沿用hugging face 的trainer的框架，考虑到需要train_dataset和eval_dataset来初始化，此处仅仅做预测
为了降低内存开销，将train.txt和dev.txt人为设置成任意的小文件，预测的结果preds_list单独写入到输出文件中
需要另外的脚本和test.txt对齐token-pred
"""
import pynvml
pynvml.nvmlInit()
# 这里的0是GPU id
def test_gpu_memory():
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return meminfo.used

def gpu_free_memory():
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return meminfo.free

from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import sys
from dataclasses import dataclass, field
from importlib import import_module
from typing import Dict, List, Optional, Tuple, TextIO
import datetime
import time
import psutil
import argparse
import logging
import math
import os
import random
import numpy as np

import torch
from torch import nn

from typing import List, Optional, Union
from transformers import PreTrainedTokenizer
from transformers import (
    AdamW,
    BertConfig,
    BertForTokenClassification,
    BertTokenizer,
    EvalPrediction,
    HfArgumentParser,
    MODEL_MAPPING,
    TrainingArguments,
    Trainer,
    set_seed,

)
sys.path.append("../train")
from utils_ner import Split, TokenClassificationDataset, TokenClassificationTask
from data import TokenClassificationDatasetGenerator

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    task_type: Optional[str] = field(
        default="NER", metadata={"help": "Task type to fine tune in training (e.g. NER, POS, etc)"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_fast: bool = field(default=False, metadata={"help": "Set this flag to use fast tokenization."})
    # If you want to tweak more attributes on your tokenizer, you should do it in a distinct script,
    # or just modify its tokenizer_config.json.
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .txt files for a CoNLL-2003-formatted task."}
    )
    labels: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a file containing all labels. If not specified, CoNLL-2003 labels are used."},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    buffer_size: int = field(
        default=1000000, metadata={"help": "max size examples to read each time ffor buffer, according to CPU volumn"}
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    elif len(sys.argv) == 3 and sys.argv[2].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[2]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) "
            f"already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    module = import_module("tasks")
    try:
        token_classification_task_clazz = getattr(module, model_args.task_type)
        token_classification_task: TokenClassificationTask = token_classification_task_clazz()
    except AttributeError:
        raise ValueError(
            f"Task {model_args.task_type} needs to be defined as a TokenClassificationTask subclass in {module}. "
            f"Available tasks classes are: {TokenClassificationTask.__subclasses__()}"
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Prepare CONLL-2003 task
    labels = token_classification_task.get_labels(data_args.labels)
    label_map: Dict[int, str] = {i: label for i, label in enumerate(labels)}
    num_labels = len(labels)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = BertConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        id2label=label_map,
        label2id={label: i for i, label in enumerate(labels)},
        cache_dir=model_args.cache_dir,
    )
    tokenizer = BertTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast,
    )

    logger.info("【耗时分析】开始载入bert模型")
    logger.info("【内存消耗】%.3fG", psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024)
    start_time = datetime.datetime.now()
    model = BertForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )
    end_time = datetime.datetime.now()
    logger.info("【耗时分析】载入bert模型耗时：%s", str(end_time-start_time))
    logger.info("【内存消耗】%.3fG", psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024)

    def align_predictions(predictions: np.ndarray, label_ids: np.ndarray) -> Tuple[List[int], List[int]]:
        preds = np.argmax(predictions, axis=2)

        batch_size, seq_len = preds.shape

        out_label_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                    out_label_list[i].append(label_map[label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])
        return preds_list, out_label_list

    def compute_metrics(p: EvalPrediction) -> Dict:
        
        preds_list, out_label_list = align_predictions(p.predictions, p.label_ids)
        return {
            "accuracy_score": accuracy_score(out_label_list, preds_list),
            "precision": precision_score(out_label_list, preds_list),
            "recall": recall_score(out_label_list, preds_list),
            "f1": f1_score(out_label_list, preds_list),
            "preds_list": preds_list,
        }


    # Get train/eval datasets
    train_dataset = (
        TokenClassificationDataset(
            token_classification_task=token_classification_task,
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.train
        )
        if training_args.do_train
        else None
    )

    eval_dataset = (
        TokenClassificationDataset(
            token_classification_task=token_classification_task,
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.dev
        )
        if training_args.do_eval
        else None
    )

    # Initialize our Trainer
    # train_dataset、eval_dataset对于预测模式没有用，可以用空文件去初始化
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )

    # 读取预测的数据
    logger.info("开始读取训练数据")
    logger.info("【内存消耗】%.3fG", psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024)
    test_file_path = os.path.join(data_args.data_dir,"test.txt")
    df_path = "/data1/aigraph/hmkg-ner/data/df_sep_by_##_oabulks.txt"
    logger.info("test file path : {}".format(test_file_path))

    # predict!
    assert training_args.do_predict==True

    model.eval()
  
    output_test_predictions_file = os.path.join(training_args.output_dir, "test_predictions.txt")
    logger.info("预测结果文件路径: {}".format(output_test_predictions_file))
    # 预测文件读取，供模型预测用
    reader = open(os.path.join(data_args.data_dir, "test.txt"), "r", encoding = "utf-8")
    # 预测时，每次读入小批量的数据，该数据量取决于CPU内存大小
    BUFFER_SIZE = data_args.buffer_size
    counter = 0
    logger.info("****************开始预测***************")
    
    preds_list_all = list()
    while True:
        test_dataset = TokenClassificationDatasetGenerator(
            token_classification_task=token_classification_task,
            data_reader=reader,
            buffer_size = BUFFER_SIZE, 
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.test
        )
        if len(test_dataset)==0:
            break
        predictions, label_ids, metrics = trainer.predict(test_dataset)
   
        #preds_list , labels_list = align_predictions(predictions, label_ids)
        preds_list = metrics.get("eval_preds_list")
        preds_list_all.extend(preds_list)
	# 注释代码只输出了label, 如果中间存在label丢弃，就无法对应上原文
        #for i in range(len(preds_list)):
        #    #预测结果写入output_test_predictions_file
        #    preds = preds_list[i]
        #    while preds:
        #        output_line = preds.pop(0) + "\n"
        #        writer.write(output_line)

        counter += 1
        logger.info("**"*50)
        logger.info("已预测完成: {}".format(BUFFER_SIZE*counter))
        logger.info("【内存消耗】%.3fG", psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024)
        logger.info("【GPU内存消耗】%.3fG", test_gpu_memory() / 1024 / 1024 / 1024)
        logger.info("【GPU可用内存】%.3fG", gpu_free_memory() / 1024 / 1024 / 1024)
        logger.info("**"*50)

    logger.info("***************结束预测************")

    reader.close()
    logger.info("预测结束")

    logger.info("***************保存预测结果到文件************")
    with open(output_test_predictions_file, 'w') as w:
        with open(os.path.join(data_args.data_dir, "test.txt"), "r", encoding = "utf-8") as r:
            write_predictions_to_file(w, r, preds_list_all)
    logger.info("***************保存预测结果到文件完成************")


def write_predictions_to_file(writer: TextIO, test_input_reader: TextIO, preds_list: List):
    example_id = 0
    for line in test_input_reader:
        line = line.strip()
        if line.startswith("==DOCSTART==") or line == "" or line == "\n":
            writer.write(line)
            if not preds_list[example_id]:
                example_id += 1
        elif preds_list[example_id]:
            output_line = line.split()[0].strip() + "\t" + preds_list[example_id].pop(0) + "\n"
            writer.write(output_line)
        else:           
            logger.warning("Maximum sequence length exceeded: No prediction for %d tokens." % len(line.split()[0]))

if __name__ == "__main__":
    main()
