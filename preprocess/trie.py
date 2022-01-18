# -*- coding: UTF-8 -*-

import pickle
import os
import json
from tqdm import tqdm
import sys
from logger import log
sys.setrecursionlimit(30000)

punctuation = {'_', '~', '(', '^', '+', '"', '#', '%', ':', '?', '`', '>', '$', '@', '*', '[', ']', '!', '&', ',',
               '{', ';', '.', '|', '}', "'", '<', '\\', '=', ')', '/'}


class TrieNode:
    def __init__(self):
        self.table = dict()
        self.phrase_end = False
        self.phrase = None

    def __len__(self):
        return len(self.table)


class TrieMatchResult:
    def __init__(self, begin, end, phrase):
        self.begin = begin
        self.end = end
        self.phrase = phrase

    def __str__(self):
        return json.dumps({
            "begin": self.begin,
            "end": self.end,
            "phrase": self.phrase,
        })

    def __repr__(self):
        return json.dumps({
            "begin": self.begin,
            "end": self.end,
            "phrase": self.phrase,
        })


class Trie:
    def __init__(self, terms_pkl_file_path, trie_pkl_file_path):
        self.root = TrieNode()
        self.terms_pkl_file_path = terms_pkl_file_path
        self.trie_pkl_file_path = trie_pkl_file_path

    def insert_phrase(self, phrase):
        node = self.root

        for ch in phrase:
            if ch not in node.table:
                node.table[ch] = TrieNode()
            node = node.table[ch]

        node.phrase_end = True
        node.phrase = phrase

    # if text[start:] has a prefix in this trie
    # return the end position of this match
    # else return -1
    def search(self, text, start=0):
        node = self.root

        tmp_result = (-1, None)
        for i in range(start, len(text)):
            if text[i] not in node.table:
                break
            node = node.table[text[i]]
            if node.phrase_end:
                if i == len(text) - 1 or text[i + 1].isspace() or text[i + 1] in punctuation:
                    tmp_result = i + 1, node.phrase
        return tmp_result

    """建Trie树"""

    def build(self):
        f = open(self.terms_pkl_file_path, "rb")
        all_terms = pickle.load(f)
        f.close()
        for term in tqdm(all_terms):
            self.insert_phrase(term)
        self.save()

    def save(self):
        log.info("saving")
        f = open(self.trie_pkl_file_path, "wb")
        pickle.dump(self.root, f)
        f.close()

    def load(self):
        log.info("loading")
        f = open(self.trie_pkl_file_path, "rb")
        self.root = pickle.load(f)
        f.close()

    """匹配文本"""

    def match(self, text):
        #text = text.lower()

        result_list = []
        i = 0
        while i < len(text):
            #if (i == 0 or text[i - 1].isspace()) and text[i].isalnum():
            if (i == 0 or text[i - 1].isspace() or text[i-1]=='(') and text[i].isalnum():
                res = self.search(text, i)
                if res[0] > 0:
                    result_item = vars(TrieMatchResult(i, res[0], res[1]))
                    result_list.append(result_item)
                    i = res[0]
                else:
                    i += 1
            else:
                i += 1
        return result_list


def gen_terms_pkl(terms_pkl_path, cleanterms_path):
    pickle_file = terms_pkl_path
    if os.path.exists(pickle_file):
        log.warning("字典文件 {} 已存在，未重新生成 ".format(pickle_file))
        return

    all_terms_file = cleanterms_path
    all_terms = dict()
    all_semantic_sty = set()
    all_sgr = set()
    with open(all_terms_file, 'r', encoding="utf-8") as reader:
        for one_line in reader:
            splits = one_line.strip().split("\t")
            cui, term, semantic_sty, sgr, upper, short_upper, cui_n, sgr_n, str_n = splits
            if cui == "cui":
                continue

            all_semantic_sty.update(semantic_sty.split("|"))
            all_sgr.update(sgr.split("|"))
            #term = term.strip().lower()
            term = term.strip()
            all_terms.setdefault(term, {"appearances": [], "semantic_sty": set(), "sgr": set()})
            all_terms[term]["appearances"].append([cui, semantic_sty, sgr, upper, short_upper])
            all_terms[term]["semantic_sty"].update(semantic_sty.split("|"))
            all_terms[term]["sgr"].update(sgr.split("|"))

    log.info("术语个数：{}".format(len(all_terms)))
    log.info("语义类型数量：{}".format(len(all_semantic_sty)))
    log.info("语义组数量：{}".format(len(all_sgr)))

    with open(pickle_file, 'wb') as writer:
        pickle.dump(all_terms, writer)
        log.info("重新生成术语字典成功")


def build_trie_tree(terms_pkl_path, trie_pkl_path):
    terms_pkl_file_path = terms_pkl_path
    trie_pkl_file_path = trie_pkl_path

    if os.path.exists(trie_pkl_file_path):
        log.warning("字典树 {} 已存在，未重新构建. ".format(trie_pkl_file_path))
        return

    trie = Trie(terms_pkl_file_path, trie_pkl_file_path)
    trie.build()
    log.info("构件字典树完成")


def match_and_tagging(mode, terms_pkl_path, trie_pkl_path, raw_text_path, tagged_text_path):
    if mode == "batch" and os.path.exists(tagged_text_path):
        log.warning("字典树匹配文件已存在，未重新匹配")
        return

    trie = Trie(terms_pkl_path, trie_pkl_path)
    trie.load()

    if mode == "test":
        while True:
            text = input("输入文本：")
            tag_result = trie.match(text)
            print("结果：%s" % tag_result)
    elif mode == "batch":
        cnt = 0
        with open(raw_text_path, encoding="utf-8") as reader:
            with open(tagged_text_path, 'w', encoding="utf-8") as writer:
                for one_line in reader:
                    cnt += 1
                    tagged_result = trie.match(one_line)
                    if len(tagged_result) == 0:
                        writer.write("[]\n")
                        continue
                    writer.write(json.dumps(tagged_result))
                    writer.write("\n")
                    if cnt % 10000 == 0:
                        log.info("进度：%d" % cnt)


if __name__ == "__main__":

    terms_path = "/platform_tech/aigraph/cleanterms/c5/cleanterms5.txt"
    terms_pkl_path = "/platform_tech/aigraph/cleanterms/c5/terms5.pkl"
    trie_pkl_path = "/platform_tech/aigraph/cleanterms/c5/trie5.pkl"

    gen_terms_pkl(terms_pkl_path, terms_path)

    log.info("try to build trie tree ...")
    build_trie_tree(terms_pkl_path, trie_pkl_path)

    match_and_tagging("test", terms_pkl_path,
                      trie_pkl_path, "", "")
