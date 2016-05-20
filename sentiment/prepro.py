import argparse
import csv
import json
import os
from collections import defaultdict, OrderedDict
import numpy as np

import itertools

import h5py
import re


def bool(string):
    return string == 'True'


def get_args():
    parser = argparse.ArgumentParser()
    home = os.path.expanduser("~")
    glove_dir = os.path.join(home, "models", "glove")
    glove_path = os.path.join(glove_dir, "glove.6B.50d.txt")
    source_dir = os.path.join(home, "data", "sst")  # Stanford Sentiment Treebank
    target_dir = os.path.join("data", "sst")
    parser.add_argument("--source_dir", type=str, default=source_dir)
    parser.add_argument("--glove_path", type=str, default=glove_path)
    parser.add_argument("--target_dir", type=str, default=target_dir)
    parser.add_argument("--skip_neutral", type=bool, default=True)

    return parser.parse_args()

def _tokenize(sentence):
    new_sentence = []
    for word in sentence:
        new_sentence.extend(re.findall(r"[\w']+", word))
    return new_sentence

def prepro(args):
    source_dir = args.source_dir
    # source paths
    str_sentences_path = os.path.join(source_dir, "SOStr.txt")
    source_scores_path = os.path.join(source_dir, "sentiment_labels.txt")
    phrases_path = os.path.join(source_dir, "dictionary.txt")
    split_path = os.path.join(source_dir, "datasetSplit.txt")

    # glove path
    glove_path = args.glove_path

    # target paths
    target_dir = args.target_dir
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    metadata_path = os.path.join(target_dir, "metadata.json")
    word2idx_path = os.path.join(target_dir, "word2idx.json")
    data_path = os.path.join(target_dir, "data.json")
    mode2ids_path = os.path.join(target_dir, "mode2ids.json")
    idx2id_path = os.path.join(target_dir, "idx2id.json")
    emb_mat_path = os.path.join(target_dir, "emb_mat.json")

    metadata = OrderedDict()
    metadata['source_dir'] = source_dir

    id2sentence_dict = OrderedDict()
    with open(str_sentences_path, 'r') as fh:
        reader = csv.reader(fh, delimiter='|')
        for id_, list_sentence in enumerate(reader):
            id_ += 1
            sentence = tuple(list_sentence)
            id2sentence_dict[id_] = sentence
    idx2id_dict = OrderedDict((idx, id_) for idx, id_ in enumerate(id2sentence_dict.keys()))
    id2idx_dict = OrderedDict((id_, idx) for idx, id_ in enumerate(id2sentence_dict.keys()))

    # word2idx dict
    def normalize(word_):
        return word_.lower()

    word_counter = defaultdict(int)
    for sentence in id2sentence_dict.values():
        sentence = _tokenize(sentence)
        for word in sentence:
            word = normalize(word)
            word_counter[word] += 1
    word_counter = OrderedDict(sorted(word_counter.items(), key=lambda item: -item[1]))
    N = 5
    print("num of distinct words: %d" % len(word_counter))
    print("top %d frequent words: %s" % (N, ", ".join(itertools.islice(word_counter.keys(), 0, N))))
    print("total num of words: %d" % sum(word_counter.values()))

    # glove
    features = OrderedDict()  # features contain glove-recognized words
    word_size = 0
    print("reading %s ... " % glove_path)
    with open(glove_path, 'r') as fh:
        for line in fh.readlines():
            array = line.lstrip().rstrip().split(" ")
            word = array[0]
            str_vector = array[1:]
            if word in word_counter:
                vector = tuple(map(float, str_vector))
                features[word] = vector
                word_size = len(vector)
    print("reading done")
    metadata['word_size'] = word_size
    unk_words = set(word_counter.keys()) - set(features.keys())
    num_unk = sum(word_counter[word] for word in unk_words)
    print("num of distinct unk words: %d" % len(unk_words))
    print("%d examples of unk words: %s" % (N, ", ".join(itertools.islice(unk_words, 0, N))))
    print("total num of unk words: %d" % num_unk)

    word2idx_dict = OrderedDict((word, idx+1) for idx, word in enumerate(sorted(features.keys())))
    idx2word_dict = OrderedDict(((idx, word) for word, idx in word2idx_dict.items()))
    vocab_size = max(word2idx_dict.values())+1
    metadata['vocab_size'] = vocab_size
    print("vocab size (including unk): %d" % vocab_size)

    """
    f = h5py.File(emb_mat_path, 'w')
    emb_mat = f.create_dataset('data', [vocab_size, word_size], dtype='float')
    for idx, word in idx2word_dict.items():
        emb_mat[idx, :] = features[word]
    """
    emb_mat = [[0] * word_size] + [features[idx2word_dict[idx]] for idx in idx2word_dict.keys()]

    UNK = "UNK"
    assert UNK not in word2idx_dict
    word2idx_dict[UNK] = 0

    def word2idx(word_):
        word_ = word_.lower()
        if word_ in word2idx_dict:
            return word2idx_dict[word_]
        else:
            return 0

    # sents
    sents = tuple(tuple(word2idx(word) for word in _tokenize(sentence)) for sentence in id2sentence_dict.values())
    max_sent_size = max(len(sent) for sent in sents)
    avg_sent_size = sum(len(sent) for sent in sents)/len(sents)
    med_sent_size = np.median([len(sent) for sent in sents])
    metadata['max_sent_size'] = max_sent_size
    metadata['avg_sent_size'] = avg_sent_size
    metadata['med_sent_size'] = med_sent_size

    print("max sent size: %d" % max_sent_size)

    pid2score_dict = OrderedDict()
    with open(source_scores_path, 'r') as fh:
        reader = csv.reader(fh, delimiter='|')
        next(reader)  # header
        for str_id, str_score in reader:
            id_ = int(str_id)
            score = float(str_score)
            pid2score_dict[id_] = score

    phrase2pid_dict = OrderedDict()
    with open(phrases_path, 'r') as fh:
        reader = csv.reader(fh, delimiter='|')
        for str_phrase, str_id in reader:
            id_ = int(str_id)
            phrase = tuple(str_phrase.split(' '))
            phrase2pid_dict[phrase] = id_

    # scores
    scores = [pid2score_dict[phrase2pid_dict[sentence]] for sentence in id2sentence_dict.values()]

    # data
    data = [sents, scores]

    # mode2ids
    if args.skip_neutral:
        print("Will skip neutral (0.4 < score < 0.6) for mode2ids dict.")
    id2mode_dict = OrderedDict()
    num2mode_dict = OrderedDict(((1, 'train'), (2, 'test'), (3, 'dev')))
    mode2ids_dict = defaultdict(list)
    with open(split_path, 'r') as fh:
        reader = csv.reader(fh, delimiter=',')
        next(reader)
        for str_id, str_num in reader:
            id_ = int(str_id)
            num = int(str_num)
            mode = num2mode_dict[num]
            if args.skip_neutral and 0.4 < scores[id2idx_dict[id_]] < 0.6:
                continue
            id2mode_dict[id_] = mode
            mode2ids_dict[mode].append(id_)

            # sanity check
            sentence = id2sentence_dict[id_]
            assert sentence in phrase2pid_dict, id_

    for mode in num2mode_dict.values():
        print("num %s examples: %d" % (mode, len(mode2ids_dict[mode])))


    # dump files
    print("dumping json files ... ")
    json.dump(metadata, open(metadata_path, 'w'))
    json.dump(word2idx_dict, open(word2idx_path, 'w'))
    json.dump(data, open(data_path, 'w'))
    json.dump(mode2ids_dict, open(mode2ids_path, 'w'))
    json.dump(idx2id_dict, open(idx2id_path, 'w'))
    json.dump(emb_mat, open(emb_mat_path, 'w'))
    print("dumping done")


def main():
    args = get_args()
    prepro(args)

if __name__ == "__main__":
    main()
