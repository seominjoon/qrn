import argparse
import os
import json
from collections import defaultdict


def bool_(string):
    if string == 'True':
        return True
    elif string == 'False':
        return False
    else:
        raise Exception()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='directed')
    parser.add_argument("--config_name", type=str, default='None')
    parser.add_argument("--task", type=str, default='1')
    parser.add_argument("--data_type", type=str, default='test')
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--data_dir", type=str, default="data/babi")
    parser.add_argument("--run_id", type=str, default="0")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--large", type=bool_, default=False)
    parser.add_argument("--trial_idx", type=str, default="1")

    args = parser.parse_args()
    return args


def summarize_result(args):
    print_accuracy_per_task(args)


def print_accuracy_per_task(args):
    model_name = args.model_name
    config_name = args.config_name.zfill(2)
    data_type = args.data_type
    data_dir = args.data_dir
    task = args.task.zfill(2)
    lang_name = args.lang + ("-10k" if args.large else "")
    run_id = args.run_id.zfill(2)
    trial_idx = args.trial_idx.zfill(2)

    target_dir = os.path.join(data_dir, lang_name, task.zfill(2))

    epoch = args.epoch
    subdir_name = "-".join([task, config_name, run_id, trial_idx])
    evals_dir = os.path.join("evals", model_name, lang_name, subdir_name)
    evals_name = "%s_%s.json" % (data_type, str(epoch).zfill(4))
    evals_path = os.path.join(evals_dir, evals_name)
    evals = json.load(open(evals_path, 'r'))

    data_path = os.path.join(target_dir, 'data.json')
    mode2idxs_path = os.path.join(target_dir, 'mode2idxs.json')
    word2idx_path = os.path.join(target_dir, 'word2idx.json')
    metadata_path = os.path.join(target_dir, 'metadata.json')
    data = json.load(open(data_path, 'r'))
    X, Q, S, Y, H, T = data
    mode2idxs_dict = json.load(open(mode2idxs_path, 'r'))
    word2idx_dict = json.load(open(word2idx_path, 'r'))
    idx2word_dict = {idx: word for word, idx in word2idx_dict.items()}
    metadata = json.load(open(metadata_path, 'r'))

    eval_dd = {}
    for idx, id_ in enumerate(evals['ids']):
        eval_d = {}
        for name, d in list(evals['values'].items()):
            eval_d[name] = d[idx]
        eval_dd[id_] = eval_d
    # id_ becomes the idx in the original data, because we don't distinguish between idx and id in babi
    # eval_dd[idx][name] = some value

    num_corrects_dict = defaultdict(int)
    total_dict = defaultdict(int)
    for id_, eval_d in eval_dd.items():
        task = T[id_]
        total_dict[task] += 1
        num_corrects_dict[task] += int(eval_d['correct'])

    acc_dict = {key: num_corrects/total_dict[key] for key, num_corrects in num_corrects_dict.items()}
    acc_pairs = sorted(acc_dict.items(), key=lambda pair: pair[0])
    for task, acc in acc_pairs:
        print("Task {0}: {1:.2f}%".format(task, acc * 100))


def main():
    args = get_args()
    summarize_result(args)


if __name__ == "__main__":
    main()
