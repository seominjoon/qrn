import json
import logging
import os
from collections import OrderedDict

import numpy as np

from config.get_config import Config


class DataSet(object):
    def __init__(self, name, batch_size, data, idxs, idx2id=None):
        self.name = name
        self.num_epochs_completed = 0
        self.idx_in_epoch = 0
        self.batch_size = batch_size
        self.data = data
        self.idxs = idxs
        if idx2id is None:
            idx2id = {idx: idx for idx in idxs}
        self.idx2id = idx2id
        self.num_examples = len(idxs)
        self.num_full_batches = int(self.num_examples / self.batch_size)
        self.num_all_batches = self.num_full_batches + int(self.num_examples % self.batch_size > 0)
        self.reset()

    def get_num_batches(self, partial=False):
        return self.num_all_batches if partial else self.num_full_batches

    def get_batch_idxs(self, partial=False):
        assert self.has_next_batch(partial=partial), "End of data, reset required."
        from_, to = self.idx_in_epoch, self.idx_in_epoch + self.batch_size
        if partial and to > self.num_examples:
            to = self.num_examples
        cur_idxs = self.idxs[from_:to]
        return cur_idxs

    def get_next_labeled_batch(self, partial=False):
        cur_idxs = self.get_batch_idxs(partial=partial)
        batch = [[each[i] for i in cur_idxs] for each in self.data]
        self.idx_in_epoch += len(cur_idxs)
        return batch

    def has_next_batch(self, partial=False):
        if partial:
            return self.idx_in_epoch < self.num_examples
        return self.idx_in_epoch + self.batch_size <= self.num_examples

    def complete_epoch(self):
        self.reset()
        self.num_epochs_completed += 1

    def reset(self):
        self.idx_in_epoch = 0
        np.random.shuffle(self.idxs)


def read_data(params, modes):
    logging.info("loading data ...")
    batch_size = params.batch_size
    data_dir = params.data_dir

    mode2ids_path = os.path.join(data_dir, "mode2ids.json")
    idx2id_path = os.path.join(data_dir, "idx2id.json")
    data_path = os.path.join(data_dir, "data.json")

    mode2ids_dict = json.load(open(mode2ids_path, 'r'))
    idx2id_dict = json.load(open(idx2id_path, 'r'))
    idx2id_dict = OrderedDict((int(idx), id_) for idx, id_ in idx2id_dict.items())
    id2idx_dict = OrderedDict((id_, idx) for idx, id_ in idx2id_dict.items())
    data = json.load(open(data_path, 'r'))
    data_sets = []
    for mode in modes:
        idxs = [id2idx_dict[id_] for id_ in mode2ids_dict[mode]]
        data_set = DataSet(mode, batch_size, data, idxs, idx2id=idx2id_dict)
        data_sets.append(data_set)
    return data_sets


def main():
    config = Config()
    config.data_dir = "data/babi-tasks"
    config.task = "1"
    config.batch_size = 100
    data_set = read_data(config, 'dev')
    print(data_set.get_num_batches(True))

if __name__ == "__main__":
    main()
