import os
import sys
import time
sys.path.append("../train")
from tasks import NER
from utils_ner import TokenClassificationDataset, Split
from typing import Dict, List, Optional, Tuple
from transformers import (
    BertConfig,
    BertForTokenClassification,
    BertTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from torch import nn
import numpy as np
from dataclasses import dataclass, field

semtype_map_path = "semtype_mapping.txt"
semtype_map = {}

with open(semtype_map_path, 'r') as r:
    _ = r.readline()
    for line in r:
        umls_sty, bios_sty = line.rstrip().split("\t")
        if bios_sty == "不要":
            continue
        semtype_map.setdefault(umls_sty, bios_sty)
    print("load semtype map ok. ", len(semtype_map))


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
        default=True, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


trainer = None
tokenizer = None
model_args, data_args, training_args = None, None, None
token_classification_task = None
labels, label_map, config = None, None, None


def load_model():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    global model_args, data_args, training_args
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    print("argument parser OK")

    global token_classification_task
    token_classification_task = NER()

    global labels
    global label_map
    labels = token_classification_task.get_labels(data_args.labels)
    label_map = {i: label for i, label in enumerate(labels)}
    num_labels = len(labels)

    global config
    config = BertConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        id2label=label_map,
        label2id={label: i for i, label in enumerate(labels)},
        cache_dir=model_args.cache_dir,
    )
    print("load BERT config OK")

    model = BertForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )
    print("load BERT model OK")

    global trainer
    trainer = Trainer(
        model=model,
        args=training_args,
    )

    global tokenizer
    tokenizer = BertTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast,
    )
    print("load BERT tokenizer OK")


def ner_predict(data_dir):
    test_dataset = TokenClassificationDataset(
        token_classification_task=token_classification_task,
        data_dir=data_dir,
        tokenizer=tokenizer,
        labels=labels,
        model_type=config.model_type,
        max_seq_length=data_args.max_seq_length,
        overwrite_cache=data_args.overwrite_cache,
        mode=Split.test,
    )

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

    def write_predictions_to_response(test_input_reader, preds_list):
        out = list()
        msg = "ok"
        for line in test_input_reader:
            line = line.strip()
            if line.startswith("==DOCSTART==") or line == "" or line == "\n":
                continue
            elif preds_list[0]:
                token = line.split()[0]
                sty = preds_list[0].pop(0)
                # print("umls sty ", sty)
                if sty.startswith("B-") or sty.startswith("I-"):
                    if sty[2:] in semtype_map:
                        sty = sty[:2] + semtype_map.get(sty[2:])
                # print("bios sty ", sty)
                out.append((token, sty))
            else:
                msg = "Maximum sequence length exceeded, drop some tokens."

        return out, msg

    predictions, label_ids, metrics = trainer.predict(test_dataset)
    print("predict done: ", len(predictions))
    preds_list, _ = align_predictions(predictions, label_ids)
    # print(preds_list)

    with open(os.path.join(data_dir, "test.txt"), 'r') as r:
        out, msg = write_predictions_to_response(r, preds_list)

    return out, msg


def generate_test(text, dir="tmp"):
    with open(os.path.join(dir, "test.txt"), 'w') as w:
        w.write("==DOCSTART==\n")
        splits = text.split()
        for word in splits:
            w.write(word + "\n")


from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/ping", methods=['GET', 'POST'])
def ping():
    return jsonify({"code":200, "msg":"ok"})

@app.route("/predict", methods=['POST'])
def predict():
    start = time.time()
    
    text = request.json["text"]
    print("receive request, data length: ", len(text))
    print("receive request, data : ", text)
    generate_test(text, "tmp_test")
    try:
        out, msg = ner_predict("tmp_test")
        return jsonify({"result": out, "msg": msg})
    except Exception as e:
        print(e)
        return jsonify({"result": None, "msg": "Model Failed"})
    finally:
        print("predict cost ", time.time() - start, " s")


if __name__ == '__main__':
    load_model()

    tmp_dir = "tmp_test"
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    # predict("/Users/xiaj/PycharmProjects/hmkg-ner/data/fix_lack_bug")
    app.run('0.0.0.0', port=8001)

