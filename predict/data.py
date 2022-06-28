"""
NerDataLoader:
为了解决训练数据读取时导致内存爆炸的问题，
每次训练时仅读取少量数据，做局部shuffle，然后yield生成。
省略中间读取所有训练数据的过程，减少内存开销
"""
from utils_ner import TokenClassificationTask, Split, InputExample, InputFeatures
from typing import List, Optional, Union, TextIO
from transformers import PreTrainedTokenizer
import random
import torch
from torch import nn
from torch.utils.data.dataset import Dataset
import os,sys

class NerDataLoader(object):
    def __init__(
        self,
        dataset_path : str, 
        instances_buffer_size : int,
        batch_size : int,
        data_helper : TokenClassificationTask, 
        tokenizer : PreTrainedTokenizer,
        labels: List[str],
        model_type: str,
        max_seq_length: int,
        shuffle:bool
    ):
        self.batch_size = batch_size
        self.instances_buffer_size = instances_buffer_size
        self.data_helper = data_helper
        self.shuffle = shuffle
        self.dataset_reader = open(dataset_path, "r")
        self.start = 0
        self.end = 0
        self.buffer = []
        self.tokenizer = tokenizer
        self.labels = labels
        self.model_type = model_type
        self.max_seq_length = max_seq_length
        
    def _fill_buf(self):
        continuous_space_line = 0
        try:
            self.buffer = []
            words = []
            labels = []
            while True:
                line = self.dataset_reader.readline()
                if line.startswith("==DOCSTART==") or line == "" or line == "\n":
                    if len(words)!=0:
                        instance = tuple([words,labels])
                        self.buffer.append(instance)
                    # 连续空行计数continuous_space_line
                    if line == "":
                        continuous_space_line += 1
                    words = []
                    labels = []
                else:
                    splits = line.split("\t")
                    words.append(splits[0])
                    if len(splits) > 1:
                        labels.append(splits[-1].replace("\n", ""))
                    else:
                        # Examples could have no label for mode = "test"
                        labels.append("O")
                # 缓存区达到最大或者dataset_reader已读到文件末
                # dataset_reader读到文件末时，不会主动退出while循环，一直会读取空行，可以通过连续空行计数continuous_space_line来判断
                if len(self.buffer) >= self.instances_buffer_size or continuous_space_line >= 2:
                    break
        except EOFError:
            # Reach file end.
            # 将指针指到文件的开头位置
            self.dataset_reader.seek(0)

        # 当前buffer的是instance即文本，将对应的features存入buffer
        self.buffer = self.featurize(self.buffer)

        if self.shuffle:
            random.shuffle(self.buffer)
        self.start = 0
        self.end = len(self.buffer)

    def _empty(self):
        return self.start >= self.end

    """
    def __del__(self):
        self.dataset_reader.close()
    """
    
    # 将instance转化为features
    def featurize(self,instances):
        #print("length of instances : {}".format(len(instances)))
        examples = self.data_helper.convert_ins_to_examples(instances)
        #print("length of examples : {}".format(len(examples)))
        #print("example 0 this batch : ")
        #print(examples[0])
        features = self.data_helper.convert_examples_to_features(
                    examples,
                    self.labels,
                    self.max_seq_length,
                    self.tokenizer,
                    cls_token_at_end=bool(self.model_type in ["xlnet"]),
                    # xlnet has a cls token at the end
                    cls_token=self.tokenizer.cls_token,
                    cls_token_segment_id=2 if self.model_type in ["xlnet"] else 0,
                    sep_token=self.tokenizer.sep_token,
                    sep_token_extra=False,
                    # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                    pad_on_left=bool(self.tokenizer.padding_side == "left"),
                    pad_token=self.tokenizer.pad_token_id,
                    pad_token_segment_id=0,
                    pad_token_label_id=-100,
                )
        #print("feature 0 this batch : ")
        #print(features[0])
        return features

    def __iter__(self):
        while True:
            print("buffer 的长度：{}".format(len(self.buffer)))
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                features = self.buffer[self.start:]
            else:
                features = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size
            print("当前start指针：{}".format(self.start))

            input_ids_list=[]
            attention_mask_list=[]
            token_type_ids_list=[]
            label_ids_list=[]
            for input_feature in features:
                input_ids_list.append(input_feature.input_ids)
                attention_mask_list.append(input_feature.attention_mask)
                token_type_ids_list.append(input_feature.token_type_ids)
                label_ids_list.append(input_feature.label_ids)
            #print(torch.LongTensor(input_ids_list).size())
            #print(torch.LongTensor(attention_mask_list).size())
            #print(torch.LongTensor(token_type_ids_list).size())
            #print(torch.LongTensor(label_ids_list).size())
            yield torch.LongTensor(input_ids_list),torch.LongTensor(attention_mask_list),torch.LongTensor(token_type_ids_list),torch.LongTensor(label_ids_list)

"""
TokenClassificationDatasetGenerator:
TokenClassificationDataset的生成器版本，在预测时使用
每次仅读取部分数据并传入CPU内存，防止内存开销过大，每次预测buffer_size个样本
"""
class TokenClassificationDatasetGenerator(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    features: List[InputFeatures]
    cached_features_file: str
    overwrite_cache: bool = False
    pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index

    # Use cross entropy ignore_index as padding label id so that only
    # real label ids contribute to the loss later.

    def __init__(
            self,
            token_classification_task: TokenClassificationTask,
            data_reader: TextIO,
            buffer_size : int,
            tokenizer: PreTrainedTokenizer,
            labels: List[str],
            model_type: str,
            max_seq_length: Optional[int] = None,
            overwrite_cache=False,
            mode: Split = Split.train,
    ):
        examples = token_classification_task.read_examples_from_reader(data_reader, buffer_size, mode)
        # TODO clean up all this to leverage built-in features of tokenizers
        self.features = token_classification_task.convert_examples_to_features(
            examples,
            labels,
            max_seq_length,
            tokenizer,
            cls_token_at_end=bool(model_type in ["xlnet"]),
            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if model_type in ["xlnet"] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=False,
            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(tokenizer.padding_side == "left"),
            pad_token=tokenizer.pad_token_id,
            pad_token_segment_id=tokenizer.pad_token_type_id,
            pad_token_label_id=self.pad_token_label_id,
        )
            
    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]
