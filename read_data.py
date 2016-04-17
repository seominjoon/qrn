import json
import os

import numpy as np

from configs.get_config import Config


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


def read_data(params, mode):
    data_dir = params.data_dir
    batch_size = params.batch_size

    print("loading {} data ... ".format(mode))

    data = None
    idxs = None
    idx2id_dict = None
    # TODO : these need to be defined. See examples below.
    """
    mode2ids_path = os.path.join(data_dir, "mode2ids.json")
    mode2ids_dict = json.load(open(mode2ids_path, 'r'))
    idx2id_path = os.path.join(data_dir, "idx2id.json")
    idx2id_dict = json.load(open(idx2id_path, 'r'))
    idx2id_dict = {int(idx): id_ for idx, id_ in idx2id_dict.items()}
    id2idx_dict = {id_: int(idx) for idx, id_ in idx2id_dict.items()}
    ids = mode2ids_dict[mode]
    idxs = [id2idx_dict[id_] for id_ in ids]

    sents_path = os.path.join(data_dir, "sents.json")
    scores_path = os.path.join(data_dir, "scores.json")
    sents = json.load(open(sents_path, 'r'))
    scores = json.load(open(scores_path, 'r'))
    data = [sents, scores]
    """

    data_set = DataSet(mode, batch_size, data, idxs, idx2id_dict)
    print("done")
    return data_set


def main():
    config = Config()
    config.data_dir = "data/mydata"
    config.batch_size = 100
    data_set = read_data(config, 'dev')
    print(data_set.get_num_batches(True))

if __name__ == "__main__":
    main()
