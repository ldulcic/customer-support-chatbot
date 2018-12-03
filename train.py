#!/usr/bin/env python3

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from dataset import dataset_factory
from model import train_model_factory
from serialization import save_object, save_model, save_vocab
from datetime import datetime
from util import embedding_size_from_name


def parse_args():
    parser = argparse.ArgumentParser(description='Script for training seq2seq chatbot.')
    parser.add_argument('--max-epochs', type=int, default=100, help='Max number of epochs models will be trained.')
    parser.add_argument('--gradient-clip', type=float, default=5, help='Gradient clip value.')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size.')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Initial learning rate.')
    parser.add_argument('--train-embeddings', action='store_true',
                        help='Should gradients be propagated to word embeddings.')
    parser.add_argument('--save-path', default='.save',
                        help='Folder where models (and other configs) will be saved during training.')
    parser.add_argument('--save-every-epoch', action='store_true',
                        help='Save model every epoch regardless of validation loss.')
    parser.add_argument('--dataset', choices=['twitter-applesupport', 'twitter-amazonhelp', 'twitter-delta',
                                              'twitter-spotifycares', 'twitter-uber_support', 'twitter-all',
                                              'twitter-small'],
                        help='Dataset for training model.')
    parser.add_argument('--teacher-forcing-ratio', type=float, default=0.5,
                        help='Teacher forcing ratio used in seq2seq models. [0-1]')

    # cuda
    gpu_args = parser.add_argument_group('GPU', 'GPU related settings.')
    gpu_args.add_argument('--cuda', action='store_true', default=False, help='Use cuda if available.')
    gpu_args.add_argument('--multi-gpu', action='store_true', default=False, help='Use multiple GPUs if available.')

    # embeddings hyperparameters
    embeddings = parser.add_mutually_exclusive_group()
    embeddings.add_argument('--embedding-type', type=str,
                            choices=['glove.42B.300d',
                                     'glove.840B.300d',
                                     'glove.twitter.27B.25d',
                                     'glove.twitter.27B.50d',
                                     'glove.twitter.27B.100d',
                                     'glove.twitter.27B.200d',
                                     'glove.6B.50d',
                                     'glove.6B.100d',
                                     'glove.6B.200d',
                                     'glove.6B.300d'],
                            help='Pre-trained embeddings type.')
    embeddings.add_argument('--embedding-size', type=int, help='Dimensionality of word embeddings.')

    # encoder hyperparameters
    encoder_args = parser.add_argument_group('Encoder', 'Encoder hyperparameters.')
    encoder_args.add_argument('--encoder-rnn-cell', choices=['LSTM', 'GRU'], default='LSTM',
                              help='Encoder RNN cell type.')
    encoder_args.add_argument('--encoder-hidden-size', type=int, default=128, help='Encoder RNN hidden size.')
    encoder_args.add_argument('--encoder-num-layers', type=int, default=1, help='Encoder RNN number of layers.')
    encoder_args.add_argument('--encoder-rnn-dropout', type=float, default=0.2, help='Encoder RNN dropout probability.')
    encoder_args.add_argument('--encoder-bidirectional', action='store_true', help='Use bidirectional encoder.')

    # decoder hyperparameters
    decoder_args = parser.add_argument_group('Decoder', 'Decoder hyperparameters.')
    decoder_args.add_argument('--decoder-type', choices=['bahdanau', 'luong'], default='bahdanau',
                              help='Type of the decoder.')
    decoder_args.add_argument('--decoder-rnn-cell', choices=['LSTM', 'GRU'], default='LSTM',
                              help='Decoder RNN cell type.')
    decoder_args.add_argument('--decoder-hidden-size', type=int, default=128, help='Decoder RNN hidden size.')
    decoder_args.add_argument('--decoder-num-layers', type=int, default=1, help='Decoder RNN number of layers.')
    decoder_args.add_argument('--decoder-rnn-dropout', type=float, default=0.2, help='Decoder RNN dropout probability.')
    decoder_args.add_argument('--luong-attn-hidden-size', type=int, default=128,
                              help='Luong decoder attention hidden projection size')
    decoder_args.add_argument('--luong-input-feed', action='store_true',
                              help='Whether Luong decoder should use input feeding approach.')
    decoder_args.add_argument('--decoder-init-type', choices=['zeros', 'bahdanau', 'adjust_pad', 'adjust_all'],
                              default='zeros', help='Decoder initial RNN hidden state initialization.')

    # attention hyperparameters
    attention_args = parser.add_argument_group('Attention', 'Attention hyperparameters.')
    attention_args.add_argument('--attention-type', choices=['none', 'global', 'local-m', 'local-p'], default='global',
                                help='Attention type.')
    attention_args.add_argument('--attention-score', choices=['dot', 'general', 'concat'], default='dot',
                                help='Attention score function type.')
    attention_args.add_argument('--half-window-size', type=int, default=5,
                                help='D parameter from Luong et al. paper. Used only for local attention.')
    attention_args.add_argument('--local-p-hidden-size', type=int, default=128,
                                help='Local-p attention hidden size (used when predicting window position).')
    attention_args.add_argument('--concat-attention-hidden-size', type=int, default=128,
                                help='Attention layer hidden size. Used only with concat score function.')

    args = parser.parse_args()

    # if none of embeddings options where given default to pre-trained glove embeddings
    if not args.embedding_type and not args.embedding_size:
        args.embedding_type = 'glove.twitter.27B.25d'

    # if embedding_size is not used, set proper pre-trained embedding size
    if not args.embedding_size:
        args.embedding_size = embedding_size_from_name(args.embedding_type)

    # add timestamp to save_path
    args.save_path += os.path.sep + datetime.now().strftime("%Y-%m-%d-%H:%M")

    print(args)
    return args


def evaluate(model, val_iter, metadata):
    model.eval()  # put models in eval mode (this is important because of dropout)

    total_loss = 0
    with torch.no_grad():
        for batch in val_iter:
            # calculate models predictions
            question, answer = batch.question, batch.answer
            logits = model(question, answer)

            # calculate batch loss
            loss = F.cross_entropy(logits.view(-1, metadata.vocab_size), answer[1:].view(-1),
                                   ignore_index=metadata.padding_idx)  # answer[1:] skip <sos> token
            total_loss += loss.item()

    return total_loss / len(val_iter)


def train(model, optimizer, train_iter, metadata, grad_clip):
    model.train()  # put models in train mode (this is important because of dropout)

    total_loss = 0
    for batch in train_iter:
        # calculate models predictions
        question, answer = batch.question, batch.answer
        logits = model(question, answer)

        # zero gradients
        optimizer.zero_grad()

        # calculate loss and backpropagate errors
        loss = F.cross_entropy(logits.view(-1, metadata.vocab_size), answer[1:].view(-1),
                               ignore_index=metadata.padding_idx)  # answer[1:] skip <sos> token
        loss.backward()

        total_loss += loss.item()

        # clip gradients to avoid exploding gradient
        clip_grad_norm_(model.parameters(), grad_clip)

        # update parameters
        optimizer.step()

    return total_loss / len(train_iter)


def main():
    args = parse_args()
    cuda = torch.cuda.is_available() and args.cuda
    torch.set_default_tensor_type(torch.cuda.FloatTensor if cuda else torch.FloatTensor)
    device = torch.device('cuda' if cuda else 'cpu')

    print("Using %s for training" % ('GPU' if cuda else 'CPU'))
    print('Loading dataset...', end='', flush=True)
    metadata, vocab, train_iter, val_iter, test_iter = dataset_factory(args, device)
    print('Done.')

    print('Saving vocab and args...', end='')
    save_vocab(vocab, args.save_path + os.path.sep + 'vocab')
    save_object(args, args.save_path + os.path.sep + 'args')
    print('Done')

    model = train_model_factory(args, metadata)
    if cuda and args.multi_gpu:
        model = nn.DataParallel(model, dim=1)  # if we were using batch_first we'd have to use dim=0
    print(model)  # print models summary

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)

    try:
        best_val_loss = None
        for epoch in range(args.max_epochs):
            start = datetime.now()
            # calculate train and val loss
            train_loss = train(model, optimizer, train_iter, metadata, args.gradient_clip)
            val_loss = evaluate(model, val_iter, metadata)
            print("[Epoch=%d/%d] train_loss %f - val_loss %f time=%s " %
                  (epoch + 1, args.max_epochs, train_loss, val_loss, datetime.now() - start), end='')

            # save models if models achieved best val loss (or save every epoch is selected)
            if args.save_every_epoch or not best_val_loss or val_loss < best_val_loss:
                print('(Saving model...', end='')
                save_model(args.save_path, model, epoch + 1, train_loss, val_loss)
                print('Done)', end='')
                best_val_loss = val_loss
            print()
    except (KeyboardInterrupt, BrokenPipeError):
        print('[Ctrl-C] Training stopped.')

    test_loss = evaluate(model, test_iter, metadata)
    print("Test loss %f" % test_loss)


if __name__ == '__main__':
    main()
