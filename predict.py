#!/usr/bin/env python3

import torch
import torch.nn as nn
import os
import argparse
from model import predict_model_factory
from dataset import field_factory, metadata_factory
from serialization import load_object
from constants import MODEL_START_FORMAT


class ModelDecorator(nn.Module):
    """
    Simple decorator around Seq2SeqPredict model which packs input question in list and unpacks output list into single
    answer. This allows client to have simple interface for dialog-like model (doesn't need to worry about wrapping and
    unwrapping).
    """

    def __init__(self, model):
        super(ModelDecorator, self).__init__()
        self.model = model

    def forward(self, question, sampling_strategy, max_seq_len):
        return self.model([question], sampling_strategy, max_seq_len)[0]


customer_service_models = {
    'apple': ('pretrained-models/apple', 39),
    'amazon': ('pretrained-models/amazon', 10),
    'uber': ('pretrained-models/uber', 58),
    'delta': ('pretrained-models/delta', 44),
    'spotify': ('pretrained-models/spotify', 14)
}


def parse_args():
    parser = argparse.ArgumentParser(description='Script for "talking" with pre-trained chatbot.')
    parser.add_argument('-cs', '--customer-service', choices=['apple', 'amazon', 'uber', 'delta', 'spotify'])
    parser.add_argument('-p', '--model-path',
                        help='Path to directory with model args, vocabulary and pre-trained pytorch models.')
    parser.add_argument('-e', '--epoch', type=int, help='Model from this epoch will be loaded.')
    parser.add_argument('--sampling-strategy', choices=['greedy', 'random', 'beam_search'], default='greedy',
                        help='Strategy for sampling output sequence.')
    parser.add_argument('--max-seq-len', type=int, default=50, help='Maximum length for output sequence.')
    parser.add_argument('--cuda', action='store_true', default=False, help='Use cuda if available.')

    args = parser.parse_args()

    if args.customer_service:
        cs = customer_service_models[args.customer_service]
        args.model_path = cs[0]
        args.epoch = cs[1]

    return args


def get_model_path(dir_path, epoch):
    name_start = MODEL_START_FORMAT % epoch
    for path in os.listdir(dir_path):
        if path.startswith(name_start):
            return dir_path + path
    raise ValueError("Model from epoch %d doesn't exist in %s" % (epoch, dir_path))


def main():
    torch.set_grad_enabled(False)
    args = parse_args()
    print('Args loaded')
    model_args = load_object(args.model_path + os.path.sep + 'args')
    print('Model args loaded.')
    vocab = load_object(args.model_path + os.path.sep + 'vocab')
    print('Vocab loaded.')

    cuda = torch.cuda.is_available() and args.cuda
    torch.set_default_tensor_type(torch.cuda.FloatTensor if cuda else torch.FloatTensor)
    print("Using %s for inference" % ('GPU' if cuda else 'CPU'))

    field = field_factory(model_args)
    field.vocab = vocab
    metadata = metadata_factory(model_args, vocab)

    model = ModelDecorator(
        predict_model_factory(model_args, metadata, get_model_path(args.model_path + os.path.sep, args.epoch), field))
    print('model loaded')
    model.eval()

    question = ''
    print('\n\nBot: Hi, how can I help you?', flush=True)
    while question != 'bye':
        while True:
            print('Me: ', end='')
            question = input()
            if question:
                break

        response = model(question, sampling_strategy=args.sampling_strategy, max_seq_len=args.max_seq_len)
        print('Bot: ' + response)


if __name__ == '__main__':
    main()
