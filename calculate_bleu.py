#!/usr/bin/env python

import torch
import pandas as pd
import os
import argparse
import subprocess
from model import predict_model_factory
from dataset import field_factory, metadata_factory
from serialization import load_object
from constants import MODEL_START_FORMAT


def parse_args():
    parser = argparse.ArgumentParser(description='Script for calculating chatbot BLEU score.')
    parser.add_argument('-p', '--model-path', required=True,
                        help='Path to directory with model args, vocabulary and pre-trained pytorch models.')
    parser.add_argument('-e', '--epoch', type=int, help='Model from this epoch will be loaded.')
    parser.add_argument('--sampling-strategy', choices=['greedy', 'random', 'beam_search'], default='greedy',
                        help='Strategy for sampling output sequence.')
    parser.add_argument('-r', '--reference-path', required=True, help='Path to reference file.')
    parser.add_argument('--max-seq-len', type=int, default=30, help='Maximum length for output sequence.')
    parser.add_argument('--cuda', action='store_true', default=False, help='Use cuda if available.')
    return parser.parse_args()


def get_model_path(dir_path, epoch):
    name_start = MODEL_START_FORMAT % epoch
    for path in os.listdir(dir_path):
        if path.startswith(name_start):
            return dir_path + path
    raise ValueError("Model from epoch %d doesn't exist in %s" % (epoch, dir_path))


def get_answers(model, questions, args):
    batch_size = 1000
    answers = []
    num_batches = len(questions) // batch_size
    rest = len(questions) % batch_size
    for batch in range(num_batches):
        batch_answers, _ = model(questions[batch * batch_size:(batch + 1) * batch_size],
                                 sampling_strategy=args.sampling_strategy,
                                 max_seq_len=args.max_seq_len)
        answers.extend(batch_answers)

    if rest != 0:
        batch_answers, _ = model(questions[-rest:], sampling_strategy=args.sampling_strategy,
                                 max_seq_len=args.max_seq_len)
        answers.extend(batch_answers)

    return answers


def main():
    torch.set_grad_enabled(False)
    args = parse_args()
    model_args = load_object(args.model_path + os.path.sep + 'args')
    vocab = load_object(args.model_path + os.path.sep + 'vocab')

    cuda = torch.cuda.is_available() and args.cuda
    torch.set_default_tensor_type(torch.cuda.FloatTensor if cuda else torch.FloatTensor)

    field = field_factory(model_args)
    field.vocab = vocab
    metadata = metadata_factory(model_args, vocab)

    model = predict_model_factory(model_args, metadata, get_model_path(args.model_path + os.path.sep, args.epoch), field)
    model.eval()

    ref = pd.read_csv(args.reference_path, sep='\t')
    ref_answers = list(map(lambda x: x.lower(), ref['answer'].tolist()))
    answers = get_answers(model, ref['question'].tolist(), args)

    with open('answers.txt', 'w') as fd:
        fd.write('\n'.join(answers))

    with open('reference.txt', 'w') as fd:
        fd.write('\n'.join(ref_answers))

    answers_file = open('answers.txt')
    bleu = subprocess.check_output("perl tools/multi-bleu.perl reference.txt", stdin=answers_file, shell=True)\
        .decode('utf-8').strip()
    answers_file.close()

    print(bleu)


if __name__ == '__main__':
    main()
