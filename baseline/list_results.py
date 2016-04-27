import json
import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bur")
    parser.add_argument("--config", default="None")
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--mode", default="dev")
    parser.add_argument("--task", default="1")
    args = parser.parse_args()
    return args

def list_results(args):
    model = args.model
    config = args.config
    mode = args.mode
    epoch = args.epoch
    task = args.task

    eval_path = os.path.join("evals", model, "%s-%s" % (config.zfill(2), task.zfill(2)),
                             "%s_%s.json" % (mode, str(epoch).zfill(4)))
    eval = json.load(open(eval_path, 'r'))
    ids = eval['ids']
    fds = eval['values']['fd']
    gds = eval['values']['gd']

    target_dir = os.path.join("data/babi", str(task).zfill(2))
    word2idx_path = os.path.join(target_dir, "word2idx.json")
    word2idx_dict = json.load(open(word2idx_path, 'r'))
    idx2word_dict = dict((idx, word) for word, idx in word2idx_dict.items())

    data_path = os.path.join(target_dir, 'data.json')
    data = json.load(open(data_path, 'r'))
    X = data[0]
    S = data[2]

    def helper(idxss):
        for idxs in idxss:
            print(" ".join([idx2word_dict[idx] for idx in idxs]))

    for fd, gd, id_ in zip(fds, gds, ids):
        x = X[id_]
        s = S[id_]

        xx = [x[ss] for ss in s]
        helper(xx)
        print()
        helper(fd)
        print()
        helper([gd])
        print("-"*10)

def main():
    args = get_args()
    list_results(args)


if __name__ == "__main__":
    main()
