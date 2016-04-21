import json
import os
import argparse

def get_args():
    pass

def list_results(model_name, config, epoch, mode, task, idx):
    eval_path = os.path.join("evals", model_name, config, "%s_%s.json" % (mode, str(epoch).zfill(4)))
    eval = json.load(open(eval_path, 'r'))
    ids = eval['ids']
    fds = eval['values']['fd']

    target_dir = os.path.join("data/babi-tasks", str(task).zfill(2))
    word2idx_path = os.path.join(target_dir, "word2idx.json")
    word2idx_dict = json.load(open(word2idx_path, 'r'))
    idx2word_dict = dict((idx, word) for word, idx in word2idx_dict.items())

    data_path = os.path.join(target_dir, 'data.json')
    data = json.load(open(data_path, 'r'))
    X = data[0]
    S = data[2]

    fd = fds[idx]
    id_ = ids[idx]
    x = X[id_]
    s = S[id_]

    def helper(idxss):
        for idxs in idxss:
            print(" ".join([idx2word_dict[idx] for idx in idxs]))

    xx = [x[ss] for ss in s]
    helper(xx)
    print()
    helper(fd)


def main():
    list_results("bur", "None", 40, "dev", 3, 1)


if __name__ == "__main__":
    main()
