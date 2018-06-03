#!/usr/bin/env python

import torch
import torch.nn as nn
import argparse
from model import predict_model_factory
from serialization import load_object


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


def parse_args():
    parser = argparse.ArgumentParser(description='Script for "talking" with pre-trained chatbot.')
    parser.add_argument('-m', '--model-path', required=True, help='Path to pre-trained pytorch model.')
    parser.add_argument('-f', '--field-path', required=True, help='Path to serialized field object.')
    parser.add_argument('-a', '--args-path', required=True, help='Path to serialized arguments object.')
    parser.add_argument('--cuda', action='store_true', default=False, help='Use cuda if available.')
    parser.add_argument('--sampling-strategy', choices=['greedy', 'random', 'beam_search'], default='greedy',
                        help='Strategy for sampling output sequence.')
    parser.add_argument('--max-seq-len', type=int, default=50, help='Maximum length for output sequence.')
    return parser.parse_args()


def main():
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    torch.set_grad_enabled(False)
    args = parse_args()
    print('parsed args')
    model_args = load_object(args.args_path)
    print('args loaded')
    field = load_object(args.field_path)
    print('field loaded')

    cuda = torch.cuda.is_available() and args.cuda
    torch.set_default_tensor_type(torch.cuda.FloatTensor if cuda else torch.FloatTensor)
    device = torch.device('cuda' if cuda else 'cpu')

    vocab_size = len(field.vocab)

    model = ModelDecorator(predict_model_factory(model_args, args.model_path, field, vocab_size))
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
