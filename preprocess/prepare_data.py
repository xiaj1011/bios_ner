#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import json
import os.path
import pickle
import random
from logger import log
from tqdm import tqdm

split_punctuation = {'–', '$', ';', '_', '}', '‐', '*', '？', '{', '°', '^', '=', '|', '-',
                     '—', ')', '~', '●', '®', '☆', '>', '“', '•', ':', '”', '+', '􀏐', '#',
                     '&', '[', '?', '`', '!', '\\', '¿', '…', '/', ']', '%', '"', '<', ',',
                     '(', "'", '@', '.'}


def split_segs(content):
    segs = content.split()
    extended_segs = []

    while True:
        for seg in segs:
            if len(seg) <= 1:
                extended_segs.append(seg)
                continue
            head_in = seg[0] in split_punctuation
            tail_in = seg[-1] in split_punctuation
            if not head_in and not tail_in:
                extended_segs.append(seg)
                continue

            if head_in:
                extended_segs.append(seg[0])
                seg = seg[1:]

            if tail_in and len(seg) > 1:
                extended_segs.extend([seg[:-1], seg[-1]])
            else:
                extended_segs.append(seg)

        if len(segs) == len(extended_segs):
            break
        segs = extended_segs
        extended_segs = []

    return extended_segs


def parse_tag_content(text, entities):
    last_end = 0
    words = []
    tags = []
    for entity in entities:
        begin = entity["begin"]
        end = entity["end"]
        # entity_type = entity.get("legal_type")  # legal_type是有问题的，合法性已经在上层通过sty或sgr判断了。
        entity_type = entity.get("predict_type")
        if begin >= last_end:
            # last_seg = text[last_end:begin].split()
            last_seg = split_segs(text[last_end:begin])
            words.extend(last_seg)
            tags.extend(["O"] * len(last_seg))

            current_seg = text[begin:end].split()
            words.extend(current_seg)
            tags.extend(["B-%s" % entity_type] + ["I-%s" % entity_type] * (len(current_seg) - 1))

            last_end = end

    # last_seg = text[last_end:].split()
    last_seg = split_segs(text[last_end:])
    words.extend(last_seg)
    tags.extend(["O"] * len(last_seg))

    # 有些段落太长了，分成几部分处理，防止token长度多于512的部分被扔掉
    ret = []
    for idx in range(0, len(words), 250):
        ret.append((words[idx:idx + 250], tags[idx:idx + 250]))
    return ret


def k_fold_samples(all_samples):
    all_sample_len = len(all_samples)
    train_count, dev_count = -1, -1
    random.shuffle(all_samples)
    train_count = int(all_sample_len * 0.9)
    dev_count = int(all_sample_len * 0.1)

    log.info("train_count {}, dev_count {}, test_count {} ".format(train_count, dev_count, all_sample_len - train_count - dev_count))
    train_samples = all_samples[:train_count]
    dev_samples = all_samples[train_count:train_count + dev_count]
    test_samples = all_samples[train_count + dev_count:]

    return train_samples, dev_samples, test_samples


def generate(raw_text_path, predict_path, terms_pkl_path,
             sty2sgr_path, dataset_dir):
    if not os.path.exists(predict_path) or not os.path.exists(raw_text_path):
        log.warning('{} 或 {} 不存在'.format(predict_path, raw_text_path))
        return

    top_concepts = {"symptom", "symptoms", "disease", "diseases", "disorder", "disorders", "syndrome", "syndromes",
                    "treatment", "treatments", "procedure", "procedures", "drug", "drugs", "medication", "medications",
                    "prevention", "preventions"}

    all_samples = []
    with open(terms_pkl_path, 'rb') as f:
        log.info("loading %s" % terms_pkl_path)
        all_terms = pickle.load(f)

    with open(sty2sgr_path, 'r', encoding="utf-8") as f:
        log.info("loading %s" % sty2sgr_path)
        sty2sgr = json.load(f)

    original_reader = open(raw_text_path, 'r', encoding="utf-8")
    predict_reader = open(predict_path, 'r', encoding="utf-8")

    num_predict = sum([1 for i in open(predict_path, "r")])
    drop_cnt = 0

    for _, one_line in tqdm(enumerate(predict_reader), total=num_predict):
        predict_line = json.loads(one_line.rstrip())
        original_line = original_reader.readline().rstrip()

        entities = []
        for entity in predict_line:
            if isinstance(entity, str):
                continue
            phrase = entity["phrase"]
            if phrase in top_concepts:
                continue

            # use = True
            ## 如果术语在字典中的appearances存在“upper=1或short.upper=1”的情况，则不标该术语
            # for appearance in all_terms[phrase]["appearances"]:
            #    if appearance[3] == "1" or appearance[4] == "1":
            #        use = False
            #        break
            # if not use:
            #    continue

            predict_type = entity["predict_type"]
            predict_sgr = sty2sgr[predict_type]

            # 注意不能使用entity['label_sty']等，因为对于修正的新术语的label_sty，消歧模型是给的随机的语义类型,Cell

            if phrase in all_terms:
                cui = all_terms[phrase]["appearances"][0][0]
                if cui.startswith("C"):  # 对于修正的新术语，直接使用消歧模型的sty
                    if predict_sgr in all_terms[phrase]["sgr"]:  # 优化
                        entities.append(entity)
                        continue
                    else:
                        drop_cnt += 1
                        continue

            # if predict_type in all_terms[phrase]["semantic_sty"]:  # 旧代码
            #    entities.append(entity)
            # elif predict_sgr in all_terms[phrase]["sgr"]:
            #    entities.append(entity)
            # else:
            #    drop_cnt += 1
            #    continue

            # entity['predict_type'] = 'Cell'  # 统一sty, 仅预测span
            # entity['predict_sgr'] = 'ANAT'

            entities.append(entity)

        ret = parse_tag_content(original_line, entities)
        all_samples.extend(ret)

    original_reader.close()
    predict_reader.close()
    log.info("原始文件全部处理完成...")
    log.info("sgr 错误，丢弃 %d " % drop_cnt)

    train_samples, dev_samples, test_samples = k_fold_samples(all_samples)

    all_labels = set()

    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)

    with open(os.path.join(dataset_dir, "train.txt"), 'w', encoding="utf-8") as writer:
        for sample in train_samples:
            writer.write("==DOCSTART==\n")
            for word, tag in zip(sample[0], sample[1]):
                writer.write("\t".join([word, tag]))
                writer.write("\n")
                all_labels.add(tag)
            writer.write("\n")
    log.info("生成train.txt完成")

    with open(os.path.join(dataset_dir, "dev.txt"), 'w', encoding="utf-8") as writer:
        for sample in dev_samples:
            writer.write("==DOCSTART==\n")
            for word, tag in zip(sample[0], sample[1]):
                writer.write("\t".join([word, tag]))
                writer.write("\n")
                all_labels.add(tag)
            writer.write("\n")
    log.info("生成dev.txt完成")

    with open(os.path.join(dataset_dir, "test.txt"), 'w', encoding="utf-8") as writer:
        for sample in test_samples:
            writer.write("==DOCSTART==\n")
            for word, tag in zip(sample[0], sample[1]):
                writer.write("\t".join([word, tag]))
                writer.write("\n")
                all_labels.add(tag)
            writer.write("\n")
    log.info("生成test.txt完成")

    with open(os.path.join(dataset_dir, "labels.txt"), 'w', encoding="utf-8") as writer:
        for label in sorted(all_labels):
            writer.write(label)
            writer.write("\n")
    log.info("生成labels.txt完成")


def generate_predict(raw_text_path, dataset_dir):
    if not os.path.exists(raw_text_path):
        log.warning('{} not exist'.format(raw_text_path))
        return

    all_samples = []

    raw_reader = open(raw_text_path, 'r', encoding="utf-8")

    num_predict = sum([1 for i in open(raw_text_path, "r")])
    for _, raw_line in tqdm(enumerate(raw_reader), total=num_predict):
        words, tags = [], []
        segs = split_segs(raw_line)
        words.extend(segs)
        tags.extend(["O"] * len(segs))
        
        for idx in range(0, len(words), 250):
            all_samples.append((words[idx:idx + 250], tags[idx:idx + 250]))
        
    raw_reader.close()
    log.info("原始文件全部处理完成...")

    test_samples = all_samples

    all_labels = set()

    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)

    with open(os.path.join(dataset_dir, "test.txt"), 'w', encoding="utf-8") as writer:
        for sample in test_samples:
            writer.write("==DOCSTART==\n")
            for word, tag in zip(sample[0], sample[1]):
                writer.write("\t".join([word, tag]))
                writer.write("\n")
                all_labels.add(tag)
            writer.write("\n")
    log.info("生成test.txt完成")


if __name__ == "__main__":
    generate()
