#!/usr/bin/env python

import torch
import argparse
from model import predict_model_factory
from serialization import load_object


def parse_args():
    parser = argparse.ArgumentParser(description='Script for "talking" with pre-trained chatbot.')
    parser.add_argument('-m', '--model-path', required=True, help='Path to pre-trained pytorch model.')
    parser.add_argument('-f', '--field-path', required=True, help='Path to serialized field object.')
    parser.add_argument('-a', '--args-path', required=True, help='Path to serialized arguments object.')
    parser.add_argument('--cuda', action='store_true', default=False, help='Use cuda if available.')
    return parser.parse_args()


def prepare_question(question, field, device):
    arr = [field.preprocess(question.lower())]
    return field.numericalize(arr, device=device)


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

    model = predict_model_factory(model_args, args.model_path, field, vocab_size)
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

        # tensor = prepare_question(question, field, device)
        #
        # token_idx = model.predict(tensor, field.vocab.stoi[SOS_TOKEN], field.vocab.stoi[EOS_TOKEN])
        # response = ' '.join(map(lambda idx: field.vocab.itos[idx], token_idx))
        response = model(question)
        print('Bot: ' + response)


if __name__ == '__main__':
    main()
