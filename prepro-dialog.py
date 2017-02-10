import argparse
import json
import logging
import os
import random
from IPython import embed
import re
from collections import OrderedDict

import itertools

EOS = "<eos>"
START = "<start>"

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
    source_dir = os.path.join(home, "data", "dialog-babi")
    parser.add_argument("--source_dir", default=source_dir)
    parser.add_argument("--target_dir", default="data/dialog-babi")
    parser.add_argument("--task", default="all")
    parser.add_argument("--dev_ratio", type=float, default=0.1)
    parser.add_argument("--use_rnn", type=bool, default = False)
    parser.add_argument("--use_match", type=bool, default = False)

    args = parser.parse_args()
    return args


def prepro(args):
    source_dir = args.source_dir
    target_dir = args.target_dir
    tasks = args.task
    dev_ratio = args.dev_ratio

    if tasks == 'all': tasks = [[str(i)] for i in range(1, 6)]
    elif tasks == 'joint': tasks = [[str(i) for i in range(1, 6)]]

    for curr_tasks in tasks:
        if args.use_rnn:
            save_tasks = ['1'+curr_task for curr_task in curr_tasks]
        elif args.use_match:
            save_tasks = ['2'+curr_task for curr_task in curr_tasks]
        else:
            save_tasks = curr_tasks

        target_parent_dir = os.path.join(target_dir, save_tasks[0].zfill(2))
        
        train_raw_data_list = []
        dev_raw_data_list = []
        test_raw_data_list = []
        test_oov_raw_data_list = []
        train_size, dev_size, test_size, test_oov_size = 0, 0, 0, 0

        for cur_task in curr_tasks:
            dstc = (cur_task.endswith('6'))

            source_train_path, source_dev_path, source_test_path, source_test_oov_path = _get_source_paths(source_dir, cur_task)

            _data =  _get_data(source_train_path, cur_task)
            train_raw_data_list.append(_data)
            _data = _get_data(source_dev_path, cur_task)
            dev_raw_data_list.append(_data)
            _data = _get_data(source_test_path, cur_task)
            test_raw_data_list.append(_data)

            if not dstc:
                _data = _get_data(source_test_oov_path, cur_task)
                test_oov_raw_data_list.append(_data)

            train_size += len(train_raw_data_list[-1][0])
            dev_size += len(dev_raw_data_list[-1][0])
            test_size += len(test_raw_data_list[-1][0])
            if not dstc:
                test_oov_size += len(test_oov_raw_data_list[-1][0])
        raw_data = [list(itertools.chain(*each)) for each in zip(*(train_raw_data_list + dev_raw_data_list + test_raw_data_list + test_oov_raw_data_list))]

        train_idxs = list(range(train_size))
        dev_idxs = list(range(train_size, train_size+dev_size))
        test_idxs = list(range(train_size+dev_size, train_size+dev_size+test_size))
        test_oov_idxs = list(range(train_size+dev_size+test_size, train_size+dev_size+test_size+test_oov_size))

        mode2idxs_dict = {'dev': dev_idxs,
                      'train': train_idxs,
                      'test': test_idxs, 'test_oov' : test_oov_idxs}
        word2idx_dicts = _get_word2idx_dict(raw_data, args.use_rnn)
        data = _apply_word2idx(word2idx_dicts, raw_data, mode2idxs_dict, args.use_rnn, args.use_match)
        if not os.path.exists(target_parent_dir):
            os.makedirs(target_parent_dir)
        _save_data(word2idx_dicts, data, target_parent_dir)
        mode2idxs_path = os.path.join(target_parent_dir, "mode2idxs.json")
        with open(mode2idxs_path, 'w') as fh: json.dump(mode2idxs_dict, fh)

def _apply_word2idx(word2idx_dicts, raw_data, idx_dict, use_rnn, use_match):

    
    w2i_dic_f, w2i_dic_a = word2idx_dicts
    paras, questions, answers, tasks = raw_data
    task = tasks[0]

    X = [[[_word2idx(w2i_dic_f, word) for word in sent] for sent in para] for para in paras]
    Q = [[_word2idx(w2i_dic_f, word) for word in ques] for ques in questions]
    Y = []

    for answer in answers:
        curr_Y = []
        for j, word in enumerate(answer):
            curr_w2i_dic_a = w2i_dic_a[0] if use_rnn else w2i_dic_a[j]
            curr_Y.append(_word2idx(curr_w2i_dic_a, word))
        Y.append(curr_Y)
    tasks = [each.zfill(2) for each in tasks]

    if not use_match:
        return [X, Q, Y, tasks]

    # In case of using match, make a list of candidate answers
    # which match with vocabs in paragraph or question

    candidates = []
    if use_rnn:
         for i in range(6): candidates.append(word2idx_dicts[1].keys())
    else:
        for w2i_dic in word2idx_dicts[1][1:]:
            candidates.append(w2i_dic.keys())

    CA = [] # A list of "all" candidates that appears in paragraph / question
    CL = [] # A list of "last" candidate that appears in paragraph / question
    num_candidate = len(candidates)

    for (x, q) in zip(paras, questions):
        sents = x + [q] # [ [], [], [], ..., [], []   ]
        words = []
        for i in sents: words += i
        words.reverse()
        candi_all, candi_last = [], []
        for _ in range(num_candidate):
            candi_all.append([])
            candi_last.append(None)
        for word in words:
            for i in range(num_candidate):
                if word in candidates[i]:
                    word_idx = word2idx_dicts[1][i+1][word]
                    if word_idx in candi_all[i]: continue
                    candi_all[i].append(word_idx)
                    if candi_last[i] is None :candi_last[i]=word_idx
        CA.append(candi_all)
        CL.append(candi_last)

    data = [X, Q, Y, CA, CL, tasks]
    return data


def _word2idx(word2idx_dict, word):
    word = _normalize(word)
    return word2idx_dict.get(word, None)


def _save_data(word2idx_dicts, data, target_dir):
    X, Q, Y = data[:3]
    max_fact_size = max(len(sent) for para in X for sent in para)
    max_ques_size = max(len(ques) for ques in Q)

    vocab_size = len(word2idx_dicts[0]), [len(dic) for dic in list(word2idx_dicts)[1]]
    metadata = {
                'max_fact_size': max_fact_size,
                'max_ques_size': max_ques_size,
                'vocab_size' : vocab_size,
                'max_sent_size': max(max_fact_size, max_ques_size),
                'max_num_sents': max(len(para) for para in X),
                'eos_idx': word2idx_dicts[0][EOS]}
    word2idx_path = os.path.join(target_dir, "word2idx.json")
    data_path = os.path.join(target_dir, "data.json")
    metadata_path = os.path.join(target_dir, "metadata.json")

    with open(word2idx_path, 'w') as fh: json.dump(word2idx_dicts, fh)
    with open(data_path, 'w') as fh: json.dump(data, fh)
    with open(metadata_path, 'w') as fh: json.dump(metadata, fh)


def _normalize(word):
    # return word.lower()
    return word


def _get_word2idx_dict(data, use_rnn):
    paras, questions, answers, _ = data
    vocab_set_fact = set(_normalize(word) for para in paras for sent in para for word in sent)
    vocab_set_fact |= set(_normalize(word) for question in questions for word in question)
    
    vocab_sets = []
    for i, answer in enumerate(answers):
        for j, word in enumerate(answer):
            if i==0:
                assert (j==len(vocab_sets))
                vocab_sets.append(set())
            vocab_sets[j].add(_normalize(word))

    for i, vocab_set in enumerate(vocab_sets[1:]):
        vocab_set.discard(None)
        if use_rnn: vocab_sets[0] |= vocab_set
    if use_rnn : vocab_sets = [vocab_sets[0]]

    word2idx_dict_fact = OrderedDict((word, idx) for idx, word in enumerate([EOS]+list(vocab_set_fact)))
    word2idx_dict_a = []
    for vocab_set in vocab_sets:
        word2idx_dict_a.append(OrderedDict((word, idx) for idx, word in enumerate(list(vocab_set))))
    return word2idx_dict_fact, word2idx_dict_a

 


def _tokenize(raw):
    tokens = re.findall(r"[\w]+", raw)
    return tokens

def _compile_ans(raw):
    words = raw.split(' ')
    phases = raw.split(':')

    # The answer type is 4.
    # (1) API_CALL
    if words[0] == 'api_call' and len(words) == 4:
        return_value = words[0], words[1], words[2], words[3], None, None, None, None
    elif words[0] == 'api_call' and len(words) == 5:
        return_value = words[0], words[1], words[2], words[3], words[4], None, None, None
    # (2) Recommendation of Restaurant
    elif phases[0] == 'what do you think of this option':
        return_value =  phases[0], None, None, None, None, phases[1][1:], None, None
    # (3) Providing extra information about restaurant
    elif raw.startswith('here it is ') and len(words) == 4:
        _list = words[-1].split('_')
        return_value =  'here it is', None, None, None, None, None, '_'.join(_list[:-1]), _list[-1]
    # (4) Default
    else:
        return_value = raw, None, None, None, None, None, None, None
    assert len(return_value) == 8
    return return_value


def _get_data(file_path, cur_task):
    paragraphs = []
    questions = []
    answers = []

    with open(file_path, 'r') as fh:
        lines = fh.readlines()
    for line_num, line in enumerate(lines):
        if line == '\n' :
            continue	
        id_, sents_ = tuple(line.lower().strip('\n').split(' ', 1))
        sents = sents_.split('\t')
        dialog = True
        if len(sents) == 1: dialog=False
        if id_ == '1': paragraph = [START]
        if dialog:
            question = _tokenize(sents[0])
            answer = _compile_ans(sents[1])
            paragraphs.append(paragraph[:])
            questions.append(question)
            answers.append(answer)
            a_ = _tokenize(answer[0])
            for phase in answer[1:]:
                if phase is not None: a_.append(phase)
            paragraph.append(a_)
        else:
            words = sents[0].split(' ')
            paragraph.append(words)

    print("Loaded %d examples" % (len(paragraphs)))
    tasks = [cur_task] * len(paragraphs)

    data = [paragraphs, questions, answers, tasks]
    return data


def _get_source_paths(source_dir, task):
    source_parent_dir = source_dir
    prefix = "dialog-babi-task%s-" % task
    train_suffix = "trn.txt"
    dev_suffix = "dev.txt"
    test_suffix = "tst.txt"
    test_oov_suffix = "tst-OOV.txt"
    names = os.listdir(source_parent_dir)
    train_name, dev_name, test_name, test_oov_name = None, None, None, None
    for name in names:
        if name.startswith(prefix):
            if name.endswith(train_suffix):
                train_name = name
            elif name.endswith(dev_suffix):
                dev_name = name
            elif name.endswith(test_suffix):
                test_name = name
            elif name.endswith(test_oov_suffix):
                test_oov_name = name
    assert train_name is not None and dev_name is not None and test_name is not None, "Invalid task number"
    train_path = os.path.join(source_parent_dir, train_name)
    dev_path = os.path.join(source_parent_dir, dev_name)
    test_path = os.path.join(source_parent_dir, test_name)
    test_oov_path = None if test_oov_name is None else os.path.join(source_parent_dir, test_oov_name)
    return train_path, dev_path, test_path, test_oov_path


def main():
    args = get_args()
    prepro(args)

if __name__ == "__main__":
    main()
