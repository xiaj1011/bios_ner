import argparse
import time
import os
import logging
import trie
import prepare_data
import argparse
import json
from logger import log

def build_trie(args):
    trie.gen_terms_pkl(args.terms_pkl_path, args.cleanterms_path)

    trie.build_trie_tree(args.terms_pkl_path, args.trie_pkl_path)


def entity_match(args):
    if not os.path.exists(args.dataset['text_path']):
        log.warning('%s not exists ' % args.dataset['text_path'])
        return

    trie.match_and_tagging("batch", args.terms_pkl_path,
                           args.trie_pkl_path, args.dataset['text_path'], args.dataset['tagged_path'])


def generate(args):
    if os.path.exists(os.path.join(args.dataset['working_dir'], 'train.txt')):
        log.warning("%s下train.txt 已经存在，不再生成train/dev/test/labels.txt文件。" % args.dataset['working_dir'])
        return

    mode = args.dataset['mode']
    if mode == 'train':
        log.info("mode: train")
        prepare_data.generate(args.dataset['text_path'],
                              args.dataset['annotated_path'], 
                              args.dataset['working_dir'])

    if mode == 'predict':
        log.info("mode: predict")
        prepare_data.generate_predict(args.dataset['text_path'], args.dataset['working_dir'])


def parse_json_args(args_json_path: str):
    args = argparse.Namespace()
    with open(args_json_path, 'r') as f:
        args.__dict__ = json.load(f)

    logging.info('Arguments:')
    for arg in vars(args):
        log.info('    {}: {}'.format(arg, json.dumps(getattr(args, arg), indent=4, ensure_ascii=False)))

    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--args_path', default='../example/configs/preprocess.json.example',
                        help='Arguments Path')
    args = parser.parse_args()
    pre_args = parse_json_args(args.args_path)

    t1 = time.time()

    log.info("step 1: trie building ...")
    build_trie(pre_args)

    mode = pre_args.dataset['mode']
    log.info(f"task mode: {mode}")    

    t2 = time.time()
    log.info("step 2: entity matching ...")
    if mode in ['train', 'trie_match']:
        entity_match(pre_args)
    else:
        log.info(f"{mode} task, SKIP entity matching.")

    t3 = time.time()
    log.info("step 3: generate train/dev/test data ...")
    if mode in ['train', 'predict', 'train_no_trie_match']:
        generate(pre_args)
    else:
        log.info(f"{mode} task, SKIP generate train/dev/test.")

    t4 = time.time()
    log.info("step 1: trie building cost {} second".format(t2 - t1))
    log.info("step 2: entity matching cost {} second" .format(t3 - t2))
    log.info("step 3: generate training data cost {} second".format(t4 - t3))

