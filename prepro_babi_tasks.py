import argparse
import json
import logging
import os

import re
from collections import OrderedDict

import itertools


def bool_(string):
    if string == "True":
        return True
    elif string == "False":
        return False
    else:
        raise Exception("Cannot cast %r to bool value." % string)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", default="~/data/babi-tasks")
    parser.add_argument("--target_dir", default="data/babi-tasks")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--task", default="1")
    parser.add_argument("--large", type=bool_, default=False)
    args = parser.parse_args()
    return args


def prepro_babi_tasks(args):
    source_dir = args.source_dir
    target_dir = args.target_dir
    lang = args.lang
    task = args.task
    is_large = args.large

    source_train_path, source_test_path = _get_source_paths(source_dir, lang, is_large, task)
    target_parent_dir = os.path.join(target_dir, lang + "-10k" if is_large else "", task.zfill(2))
    train_raw_data = _get_data(source_train_path)
    test_raw_data = _get_data(source_test_path)
    raw_data = [list(itertools.chain(*each)) for each in zip(train_raw_data, test_raw_data)]
    train_size, test_size = len(train_raw_data[0]), len(test_raw_data[0])
    mode2idxs_dict = {'train': list(range(train_size)),
                      'test': list(range(train_size, train_size + test_size))}
    word2idx_dict = _get_word2idx_dict(train_raw_data)
    data = _apply_word2idx(word2idx_dict, raw_data)
    if not os.path.exists(target_parent_dir):
        os.makedirs(target_parent_dir)
    _save_data(word2idx_dict, data, target_parent_dir)
    mode2idxs_path = os.path.join(target_parent_dir, "mode2idxs.json")
    json.dump(mode2idxs_dict, open(mode2idxs_path, 'w'))


def _apply_word2idx(word2idx_dict, raw_data):
    paras, questions, S, answers = raw_data
    X = [[[_word2idx(word2idx_dict, word) for word in sent] for sent in para] for para in paras]
    Q = [[_word2idx(word2idx_dict, word) for word in ques] for ques in questions]
    Y = [_word2idx(word2idx_dict, word) for word in answers]
    data = [X, Q, S, Y]
    return data


def _word2idx(word2idx_dict, word):
    word = _normalize(word)
    return word2idx_dict[word] if word in word2idx_dict else 0


def _save_data(word2idx_dict, data, target_dir):
    X, Q, S, Y = data
    metadata = {'vocab_size': len(word2idx_dict),
                'max_sent_size': max(len(sent) for para in X for sent in para),
                'max_ques_size': max(len(ques) for ques in Q),
                'max_num_sents': max(len(para) for para in X)}
    word2idx_path = os.path.join(target_dir, "word2idx.json")
    data_path = os.path.join(target_dir, "data.json")
    metadata_path = os.path.join(target_dir, "metadata.json")
    json.dump(word2idx_dict, open(word2idx_path, 'w'))
    json.dump(data, open(data_path, 'w'))
    json.dump(metadata, open(metadata_path, 'w'))


def _normalize(word):
    return word.lower()


def _get_word2idx_dict(data):
    paras, questions, supports, answers = data
    vocab_set = set(_normalize(word) for para in paras for sent in para for word in sent)
    vocab_set |= set(_normalize(word) for question in questions for word in question)
    word2idx_dict = OrderedDict((word, idx + 1) for idx, word in enumerate(list(vocab_set)))
    word2idx_dict['UNK'] = 0
    return word2idx_dict


def _tokenize(raw):
    tokens = re.findall(r"[\w]+", raw)
    return tokens


_s_re = re.compile("^(\\d+) ([\\w\\s.]+)")
_q_re = re.compile("^(\\d+) ([\\w\\s\\?]+)\t([\\w,]+)\t([\\d+ ]+)")


def _get_data(file_path):
    paragraphs = []
    questions = []
    supports = []
    answers = []

    with open(file_path, 'r') as fh:
        lines = fh.readlines()
    paragraph = []
    for line_num, line in enumerate(lines):
        sm = _s_re.match(line)
        qm = _q_re.match(line)
        if qm:
            id_, raw_question, answer, raw_support = qm.groups()
            question = _tokenize(raw_question)
            paragraphs.append(paragraph[:])
            questions.append(question)
            answers.append(answer)
            support = [int(str_num) - 1 for str_num in raw_support.split(" ")]
            supports.append(support)
        elif sm:
            id_, raw_sentence = sm.groups()
            sentence = _tokenize(raw_sentence)
            if id_ == '1':
                paragraph = []
            paragraph.append(sentence)
        else:
            logging.error("Line %d is invalid at %s." % (line_num + 1, file_path))
    print("Loaded %d examples from %s" % (len(paragraphs), os.path.basename(file_path)))

    data = [paragraphs, questions, supports, answers]
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
    prepro_babi_tasks(args)

if __name__ == "__main__":
    main()
