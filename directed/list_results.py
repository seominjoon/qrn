import shutil
from collections import OrderedDict
import http.server
import socketserver
import argparse
import json
import os
import numpy as np

from jinja2 import Environment, FileSystemLoader

from my.utils import get_pbar


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='directed')
    parser.add_argument("--config_name", type=str, default='None')
    parser.add_argument("--task", type=str, default='1')
    parser.add_argument("--data_type", type=str, default='test')
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--template_name", type=str, default="list_results.html")
    parser.add_argument("--num_per_page", type=int, default=1000)
    parser.add_argument("--data_dir", type=str, default="data/babi")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--open", type=str, default='False')

    args = parser.parse_args()
    return args


def _decode(decoder, sent):
    return " ".join(decoder[idx] for idx in sent)


def list_results(args):
    model_name = args.model_name
    config_name = args.config_name
    data_type = args.data_type
    num_per_page = args.num_per_page
    data_dir = args.data_dir
    task = args.task

    target_dir = os.path.join(data_dir, task.zfill(2))

    epoch = args.epoch
    evals_dir = os.path.join("evals", model_name, "{}-{}".format(config_name.zfill(2), task.zfill(2)))
    evals_name = "%s_%s.json" % (data_type, str(epoch).zfill(4))
    evals_path = os.path.join(evals_dir, evals_name)
    evals = json.load(open(evals_path, 'r'))


    _id = 0
    html_dir = "/tmp/list_results%d" % _id
    while os.path.exists(html_dir):
        _id += 1
        html_dir = "/tmp/list_results%d" % _id

    if os.path.exists(html_dir):
        shutil.rmtree(html_dir)
    os.mkdir(html_dir)

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    templates_dir = os.path.join(cur_dir, 'templates')
    env = Environment(loader=FileSystemLoader(templates_dir))
    env.globals.update(zip=zip, reversed=reversed)
    template = env.get_template(args.template_name)

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

    rows = []
    pbar = get_pbar(len(eval_dd)).start()
    for i, (id_, eval_d) in enumerate(eval_dd.items()):
        question = _decode(idx2word_dict, Q[id_])
        facts = [_decode(idx2word_dict, x) for x in X[id_]]
        correct = eval_d['correct']
        a_raw = np.transpose(eval_d['a'])  # [M, L]
        a = [["%.2f" % val for val in l] for l in a_raw]
        of_raw = np.transpose(eval_d['of'])  # [M, L]
        of = [["%.2f" % val for val in l] for l in of_raw]
        ob_raw = np.transpose(eval_d['ob'])  # [M, L]
        ob = [["%.2f" % val for val in l] for l in ob_raw]
        row = {'id': id_,
               'facts': facts,
               'question': question,
               'a': a,
               'of': of,
               'ob': ob,
               'num_layers': len(a[0]),
               'correct': correct,
               'task': T[i],
               'y': idx2word_dict[Y[id_]],
               'yp': idx2word_dict[eval_d['yp']]}
        rows.append(row)

        if i % num_per_page == 0:
            html_path = os.path.join(html_dir, "%s.html" % str(id_).zfill(8))

        if (i + 1) % num_per_page == 0 or (i + 1) == len(eval_dd):
            var_dict = {'title': "Sentence List",
                        'rows': rows
                        }
            with open(html_path, "wb") as f:
                f.write(template.render(**var_dict).encode('UTF-8'))
            rows = []
        pbar.update(i)
    pbar.finish()

    os.chdir(html_dir)
    port = args.port
    host = args.host
    # Overriding to suppress log message
    class MyHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass
    handler = MyHandler
    httpd = socketserver.TCPServer((host, port), handler)
    if args.open == 'True':
        os.system("open http://%s:%d" % (args.host, args.port))
    print("serving at %s:%d" % (host, port))
    httpd.serve_forever()


if __name__ == "__main__":
    ARGS = get_args()
    list_results(ARGS)