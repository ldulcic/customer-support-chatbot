#!/usr/bin/env python

import sys
import torch
import argparse
from model import model_factory
from serialization import load_object
from constants import SOS_TOKEN, EOS_TOKEN, CUDA
from util import cuda


def parse_args():
    parser = argparse.ArgumentParser(description='Script for "talking" with pre-trained chatbot.')
    parser.add_argument('-m', '--model-path', required=True, help='Path to pre-trained pytorch model.')
    parser.add_argument('-f', '--field-path', required=True, help='Path to serialized field object.')
    parser.add_argument('-a', '--args-path', required=True, help='Path to serialized arguments object.')
    return parser.parse_args()


def prepare_question(question, field):
    arr = [field.preprocess(question.lower())]
    return field.numericalize(arr, device=0 if CUDA else -1)


def main():
    torch.set_grad_enabled(False)
    args = parse_args()
    field = load_object(args.field_path)
    model_args = load_object(args.args_path)

    vocab_size = len(field.vocab)

    model = cuda(model_factory(model_args, field, vocab_size, field.vocab.stoi['<pad>']))
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    question = ''
    print('Hi, how can I help you?')
    while question != 'bye':
        question = input()

        tensor = cuda(prepare_question(question, field))

        token_idx = model.predict(tensor, field.vocab.stoi[SOS_TOKEN], field.vocab.stoi[EOS_TOKEN])
        response = ' '.join(map(lambda idx: field.vocab.itos[idx], token_idx))
        print(response)


if __name__ == '__main__':
    main()
