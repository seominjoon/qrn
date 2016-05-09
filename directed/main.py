import json
import os
import shutil
from pprint import pprint

import tensorflow as tf

from directed.model import Tower, Runner
from configs.get_config import get_config_from_file, get_config
from directed.read_data import read_data, read_one_data

flags = tf.app.flags

# File directories
flags.DEFINE_string("model_name", "directed", "Model name. This will be used for save, log, and eval names. [directed]")
flags.DEFINE_string("data_dir", "data/babi", "Data directory [data/babi]")

# Training parameters
flags.DEFINE_integer("batch_size", 32, "Batch size for each tower. [32]")
flags.DEFINE_float("init_mean", 0, "Initial weight mean [0]")
flags.DEFINE_float("init_std", 0.1, "Initial weight std [0.1]")
flags.DEFINE_float("init_lr", 0.5, "Initial learning rate [0.5]")
flags.DEFINE_integer("lr_anneal_period", 20, "Anneal period [20]")
flags.DEFINE_float("lr_anneal_ratio", 0.5, "Anneal ratio [0.5")
flags.DEFINE_integer("num_epochs", 100, "Total number of epochs for training [100]")
flags.DEFINE_string("opt", 'basic', 'Optimizer: basic | adagrad | adam [basic]')

# Training and testing options
flags.DEFINE_boolean("train", False, "Train? Test if False [False]")
flags.DEFINE_integer("val_num_batches", 0, "Val num batches. 0 for max possible. [0]")
flags.DEFINE_integer("train_num_batches", 0, "Train num batches. 0 for max possible [0]")
flags.DEFINE_integer("test_num_batches", 0, "Test num batches. 0 for max possible [0]")
flags.DEFINE_boolean("load", False, "Load from saved model? [False]")
flags.DEFINE_boolean("progress", True, "Show progress bar? [True]")
flags.DEFINE_string("device_type", 'cpu', "cpu | gpu [cpu]")
flags.DEFINE_integer("num_devices", 1, "Number of devices to use. Only for multi-GPU. [1]")
flags.DEFINE_integer("val_period", 5, "Validation period (for display purpose only) [5]")
flags.DEFINE_integer("save_period", 10, "Save period [10]")
flags.DEFINE_string("config", 'None', "Config name (e.g. local) to load. 'None' to use configs here. [None]")
flags.DEFINE_string("config_ext", ".json", "Config file extension: .json | .tsv [.json]")

# Debugging
flags.DEFINE_boolean("draft", False, "Draft? (quick initialize) [False]")

# App-specific options
# TODO : Any other options
flags.DEFINE_string("task", "all", "Task number. [all]")
flags.DEFINE_integer("hidden_size", 50, "Hidden size. [50]")
flags.DEFINE_integer("max_grad_norm", 0, "Max grad norm. 0 for no clipping [0]")
flags.DEFINE_integer("rnn_num_layers", 1, "RNN number of layers [1]")
flags.DEFINE_float("keep_prob", 0.5, "Keep probability of RNN inputs [0.5]")
flags.DEFINE_integer("mem_num_layers", 2, "Number of memory layers [2]")
flags.DEFINE_float("wd", 0, "Weight decay [0]")
flags.DEFINE_float("forget_bias", 2.5, "Forget bias [2.5]")

FLAGS = flags.FLAGS


def mkdirs(config):
    evals_dir = "evals"
    logs_dir = "logs"
    saves_dir = "saves"
    if not os.path.exists(evals_dir):
        os.mkdir(evals_dir)
    if not os.path.exists(logs_dir):
        os.mkdir(logs_dir)
    if not os.path.exists(saves_dir):
        os.mkdir(saves_dir)

    eval_dir = os.path.join(evals_dir, config.model_name)
    eval_subdir = os.path.join(eval_dir, "%s-%s" % (str(config.config).zfill(2), config.task.zfill(2)))
    log_dir = os.path.join(logs_dir, config.model_name)
    log_subdir = os.path.join(log_dir, "%s-%s" % (str(config.config).zfill(2), config.task.zfill(2)))
    save_dir = os.path.join(saves_dir, config.model_name)
    save_subdir = os.path.join(save_dir, "%s-%s" % (str(config.config).zfill(2), config.task.zfill(2)))
    config.eval_dir = eval_subdir
    config.log_dir = log_subdir
    config.save_dir = save_subdir

    if not os.path.exists(eval_dir):
        os.mkdir(eval_dir)
    if os.path.exists(eval_subdir):
        if config.train and not config.load:
            shutil.rmtree(eval_subdir)
            os.mkdir(eval_subdir)
    else:
        os.mkdir(eval_subdir)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if os.path.exists(log_subdir):
        if config.train and not config.load:
            shutil.rmtree(log_subdir)
            os.mkdir(log_subdir)
    else:
        os.mkdir(log_subdir)
    if config.train:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        if os.path.exists(save_subdir):
            if not config.load:
                shutil.rmtree(save_subdir)
                os.mkdir(save_subdir)
        else:
            os.mkdir(save_subdir)


def load_meta_data(config):
    metadata_path = os.path.join(config.data_dir, config.task.zfill(2), "metadata.json")
    metadata = json.load(open(metadata_path, "r"))

    # TODO: set other parameters, e.g.
    # config.max_sent_size = meta_data['max_sent_size']
    config.max_fact_size = metadata['max_fact_size']
    config.max_ques_size = metadata['max_ques_size']
    config.max_sent_size = metadata['max_sent_size']
    config.vocab_size = metadata['vocab_size']
    config.max_num_sents = metadata['max_num_sents']
    config.max_num_sups = metadata['max_num_sups']
    config.eos_idx = metadata['eos_idx']


def main(_):
    if FLAGS.config == "None":
        config = get_config(FLAGS.__flags, {})
    else:
        # TODO : create configs file (.json)
        config_path = os.path.join("configs", "%s%s" % (FLAGS.model_name, FLAGS.config_ext))
        config = get_config_from_file(FLAGS.__flags, config_path, FLAGS.config)

    load_meta_data(config)
    mkdirs(config)

    # load other files
    if config.train:
        comb_train_ds = read_one_data(config, 'train', config.task)
        comb_dev_ds = read_one_data(config, 'dev', config.task)
    else:
        comb_test_ds = read_one_data(config, 'test', config.task)

    # For quick draft initialize (deubgging).
    if config.draft:
        config.train_num_batches = 1
        config.val_num_batches = 1
        config.test_num_batches = 1
        config.num_epochs = 2
        config.val_period = 1
        config.save_period = 1
        # TODO : Add any other parameter that induces a lot of computations

    pprint(config.__dict__)

    # TODO : specify eval tensor names to save in evals folder
    eval_tensor_names = ['correct', 'yp']

    graph = tf.Graph()
    # TODO : initialize BaseTower-subclassed objects
    towers = [Tower(config) for _ in range(config.num_devices)]
    sess = tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True))
    # TODO : initialize BaseRunner-subclassed object
    runner = Runner(config, sess, towers)
    with graph.as_default(), tf.device("/cpu:0"):
        runner.initialize()
        if config.train:
            if config.load:
                runner.load()
            runner.train(comb_train_ds, config.num_epochs, val_data_set=comb_dev_ds,
                         eval_tensor_names=eval_tensor_names, num_batches=config.train_num_batches,
                         val_num_batches=config.val_num_batches)
        else:
            runner.load()
            runner.eval(comb_test_ds, eval_tensor_names=eval_tensor_names,
                        num_batches=config.test_num_batches)


if __name__ == "__main__":
    tf.app.run()
