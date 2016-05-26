import argparse
import os
import json
import csv

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
    source_dir = os.path.join(home, "data", "MCTest")
    parser.add_argument("--source_dir", default=source_dir)
    parser.add_argument("--target_dir", default="data/mctest")
    parser.add_argument("--size", default="160")
    args = parser.parse_args()
    return args


def get_story_question_pairs(tsv_path):
    with open(tsv_path, 'r') as fh:
        pairs = []
        reader = csv.reader(fh, delimiter='\t')
        for row in reader:
            id_, properties, story = row[:3]
            questions = row[3:]
            for i in range(0, len(questions), 5):
                question = questions[i]
                choices = questions[i+1:i+5]
                pairs.append([story, question, choices])
    return pairs


def get_answers(ans_path):
    with open(ans_path, 'r') as fh:
        reader = csv.reader(fh, delimiter='\t')
        answers = list(reader)
    return answers


def prepro(args):
    source_dir = args.source_dir
    target_dir = args.target_dir
    size = args.size
    data_types = ('train', 'dev', 'test')
    tsv_paths = [os.path.join(source_dir, "mc{}.{}.tsv".format(size, type_)) for type_ in data_types]
    ans_paths = [os.path.join(source_dir, "mc{}.{}.ans".format(size, type_)) for type_ in data_types]
    # [[[story, question, choices, answer], [], ...], [], []]
    raw_data = [[pair + [answer] for pair, answer in zip(get_story_question_pairs(tsv_path), get_answers(ans_path))]
                for tsv_path, ans_path in zip(tsv_paths, ans_paths)]
