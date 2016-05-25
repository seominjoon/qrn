import argparse
import json
import logging
import os
import random

import re
from collections import OrderedDict

import itertools

from qa2hypo import qa2hypo

EOS = "<eos>"


def bool_(string):
    if string == "True":
        return True
    elif string == "False":
        return False
    else:
        raise Exception("Cannot cast %r to bool value." % string)


def get_args():
    parser = argparse.ArgumentParser()
    home = os.path.expanduser("~")
    source_dir = os.path.join(home, "data", "babi")
    parser.add_argument("--source_dir", default=source_dir)
    parser.add_argument("--target_dir", default="data/babi")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--task", default="1")
    parser.add_argument("--large", type=bool_, default=False)
    parser.add_argument("--dev_ratio", type=float, default=0.1)
    args = parser.parse_args()
    return args


def prepro(args):
    source_dir = args.source_dir
    target_dir = args.target_dir
    lang = args.lang
    task = args.task
    is_large = args.large
    dev_ratio = args.dev_ratio

    all_tasks = list(map(str, range(1, 21)))
    tasks = all_tasks if task == 'all' else task.split(",")
    target_parent_dir = os.path.join(target_dir, lang + ("-10k" if is_large else ""), task.zfill(2))
    train_raw_data_list = []
    test_raw_data_list = []
    train_size, test_size = 0, 0

    for cur_task in tasks:
        source_train_path, source_test_path = _get_source_paths(source_dir, lang, is_large, cur_task)
        train_raw_data_list.append(_get_data(source_train_path, cur_task))
        test_raw_data_list.append(_get_data(source_test_path, cur_task))
        train_size += len(train_raw_data_list[-1][0])
        test_size += len(test_raw_data_list[-1][0])

    raw_data = [list(itertools.chain(*each)) for each in zip(*(train_raw_data_list + test_raw_data_list))]
    dev_size = int(train_size * dev_ratio)
    dev_idxs = sorted(random.sample(list(range(train_size)), dev_size))
    train_idxs = [a for a in range(train_size) if a not in dev_idxs]
    test_idxs = list(range(train_size, train_size + test_size))

    mode2idxs_dict = {'dev': dev_idxs,
                      'train': train_idxs,
                      'test': test_idxs}
    word2idx_dict = _get_word2idx_dict(raw_data)
    data = _apply_word2idx(word2idx_dict, raw_data)
    if not os.path.exists(target_parent_dir):
        os.makedirs(target_parent_dir)
    _save_data(word2idx_dict, data, target_parent_dir)
    mode2idxs_path = os.path.join(target_parent_dir, "mode2idxs.json")
    with open(mode2idxs_path, 'w') as fh: json.dump(mode2idxs_dict, fh)


def _apply_word2idx(word2idx_dict, raw_data):
    paras, questions, S, answers, hypos, tasks = raw_data
    X = [[[_word2idx(word2idx_dict, word) for word in sent] for sent in para] for para in paras]
    Q = [[_word2idx(word2idx_dict, word) for word in ques] for ques in questions]
    Y = [_word2idx(word2idx_dict, word) for word in answers]
    H = [[_word2idx(word2idx_dict, word) for word in hypo] for hypo in hypos]
    tasks = [each.zfill(2) for each in tasks]
    data = [X, Q, S, Y, H, tasks]
    return data


def _word2idx(word2idx_dict, word):
    word = _normalize(word)
    return word2idx_dict[word]


def _save_data(word2idx_dict, data, target_dir):
    X, Q, S, Y, H, T = data
    max_fact_size = max(len(sent) for para in X for sent in para)
    max_ques_size = max(len(ques) for ques in Q)
    max_hypo_size = max(len(hypo) for hypo in H)
    metadata = {'vocab_size': len(word2idx_dict),
                'max_fact_size': max_fact_size,
                'max_ques_size': max_ques_size,
                'max_hypo_size': max_hypo_size,
                'max_sent_size': max(max_fact_size, max_ques_size, max_hypo_size),
                'max_num_sents': max(len(para) for para in X),
                'max_num_sups': max(len(sups) for sups in S),
                'eos_idx': word2idx_dict[EOS]}
    word2idx_path = os.path.join(target_dir, "word2idx.json")
    data_path = os.path.join(target_dir, "data.json")
    metadata_path = os.path.join(target_dir, "metadata.json")
    with open(word2idx_path, 'w') as fh: json.dump(word2idx_dict, fh)
    with open(data_path, 'w') as fh: json.dump(data, fh)
    with open(metadata_path, 'w') as fh: json.dump(metadata, fh)


def _normalize(word):
    # return word.lower()
    return word


def _get_word2idx_dict(data):
    paras, questions, supports, answers, hypos, tasks = data
    vocab_set = set(_normalize(word) for para in paras for sent in para for word in sent)
    vocab_set |= set(_normalize(word) for question in questions for word in question)
    vocab_set |= set(_normalize(word) for word in answers)
    vocab_set |= set(_normalize(word) for hypo in hypos for word in hypo)
    # Add other vocabs
    vocab_set.add(EOS)

    word2idx_dict = OrderedDict((word, idx) for idx, word in enumerate(list(vocab_set)))
    return word2idx_dict


def _tokenize(raw):
    tokens = re.findall(r"[\w]+", raw)
    return tokens


_s_re = re.compile("^(\\d+) ([\\w\\s.]+)")
_q_re = re.compile("^(\\d+) ([\\w\\s\\?]+)\t([\\w,]+)\t([\\d+ ]+)")


def _get_data(file_path, cur_task):
    paragraphs = []
    questions = []
    supports = []
    answers = []
    hypos = []

    with open(file_path, 'r') as fh:
        lines = fh.readlines()
    paragraph = []
    num2idx_dict = {}
    for line_num, line in enumerate(lines):
        sm = _s_re.match(line)
        qm = _q_re.match(line)
        if qm:
            id_, raw_question, answer, raw_support = qm.groups()
            question = _tokenize(raw_question)
            raw_hypo = qa2hypo(raw_question, answer)
            hypo = _tokenize(raw_hypo)
            paragraphs.append(paragraph[:])
            questions.append(question)
            answers.append(answer)
            hypos.append(hypo)
            support = [num2idx_dict[str_num] for str_num in raw_support.split(" ")]
            supports.append(support)
        elif sm:
            id_, raw_sentence = sm.groups()
            sentence = _tokenize(raw_sentence)
            if id_ == '1':
                paragraph = []
                num2idx_dict = {}
            num2idx_dict[id_] = len(paragraph)
            paragraph.append(sentence)
        else:
            logging.error("Line %d is invalid at %s." % (line_num + 1, file_path))
    print("Loaded %d examples from %s" % (len(paragraphs), os.path.basename(file_path)))
    tasks = [cur_task] * len(paragraphs)

    data = [paragraphs, questions, supports, answers, hypos, tasks]
    return data


def _get_source_paths(source_dir, lang, is_large, task):
    source_parent_dir = os.path.join(source_dir, lang + ("-10k" if is_large else ""))
    prefix = "qa%s_" % task
    train_suffix = "train.txt"
    test_suffix = "test.txt"
    names = os.listdir(source_parent_dir)
    train_name, test_name = None, None
    for name in names:
        if name.startswith(prefix):
            if name.endswith(train_suffix):
                train_name = name
            elif name.endswith(test_suffix):
                test_name = name
    assert train_name is not None and test_name is not None, "Invalid task number"
    train_path = os.path.join(source_parent_dir, train_name)
    test_path = os.path.join(source_parent_dir, test_name)
    return train_path, test_path


def main():
    args = get_args()
    prepro(args)

if __name__ == "__main__":
    main()
