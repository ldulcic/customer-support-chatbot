import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import dill
import argparse
from torch.nn.utils import clip_grad_norm_
from dataset import dataset_factory
from model.model import Encoder, Decoder, Seq2Seq
from util import cuda, embedding_size_from_name
from constants import CUDA


def parse_args():
    parser = argparse.ArgumentParser(description='Script for training seq2seq chatbot.')
    parser.add_argument('--max-epochs', type=int, default=100, help='Max number of epochs model will be trained.')
    parser.add_argument('--gradient-clip', type=float, default=5, help='Gradient clip value.')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size.')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Initial learning rate.')
    parser.add_argument('--train-embeddings', action='store_true',
                        help='Should gradients be propagated to word embeddings.')

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
    parser.add_argument('--encoder-hidden-size', type=int, default=128, help='Encoder RNN hidden size.')
    parser.add_argument('--encoder-num-layers', type=int, default=1, help='Encoder RNN number of layers.')
    parser.add_argument('--encoder-bidirectional', action='store_true', help='Use bidirectional encoder.')

    # decoder hyperparameters
    parser.add_argument('--decoder-hidden-size', type=int, default=128, help='Decoder RNN hidden size.')
    parser.add_argument('--decoder-num-layers', type=int, default=1, help='Decoder RNN number of layers.')

    args = parser.parse_args()

    # if none of embeddings options where given default to pre-trained glove embeddings
    if not args.embedding_type and not args.embedding_size:
        args.embedding_type = 'glove.twitter.27B.25d'

    # if embedding_size is not used, set proper pre-trained embedding size
    if not args.embedding_size:
        args.embedding_size = embedding_size_from_name(args.embedding_type)

    print(args)
    return args


def save_model(model, epoch, val_loss, field):
    if not os.path.isdir('.save'):
        os.makedirs('.save')
    torch.save(model.state_dict(), ".save/seq2seq-%d-%f.pt" % (epoch, val_loss))
    dill.dump(field, open('.save/field', 'wb'))


def evaluate(model, val_iter, vocab_size, padding_idx):
    model.eval()  # put model in eval mode (this is important because of dropout)

    total_loss = 0
    for batch in val_iter:
        # calculate model predictions
        question, answer = cuda(batch.question), cuda(batch.answer)
        outputs = model(question, answer)

        # calculate batch loss
        loss = F.nll_loss(outputs.view(-1, vocab_size), answer[1:].view(-1),
                          ignore_index=padding_idx)  # answer[1:] skip <sos> token
        total_loss += loss.item()

    return total_loss / len(val_iter)


def train(model, optimizer, train_iter, vocab_size, grad_clip, padding_idx):
    model.train()  # put model in train mode (this is important because of dropout)

    optimizer.zero_grad()
    total_loss = 0
    for batch in train_iter:
        # calculate model predictions
        question, answer = cuda(batch.question), cuda(batch.answer)
        outputs = model(question, answer)

        # calculate loss and backpropagate errors
        loss = F.nll_loss(outputs.view(-1, vocab_size), answer[1:].view(-1),
                          ignore_index=padding_idx)  # answer[1:] skip <sos> token
        loss.backward()

        total_loss += loss.item()

        # clip gradients to avoid exploding gradient
        clip_grad_norm_(model.parameters(), grad_clip)

        # update parameters
        optimizer.step()

    return total_loss / len(train_iter)


def main():
    print("Using %s for training..." % ('GPU' if CUDA else 'CPU'))
    args = parse_args()
    field, train_iter, val_iter, test_iter = dataset_factory('twitter-customer-support', args)

    vocab_size = len(field.vocab)
    padding_idx = field.vocab.stoi['<pad>']

    # init encoder and decoder
    encoder = Encoder(vocab_size=vocab_size, embed_size=args.embedding_size, hidden_size=args.encoder_hidden_size,
                      num_layers=args.encoder_num_layers, bidirectional=args.encoder_bidirectional)
    decoder = Decoder(vocab_size=vocab_size, embed_size=args.embedding_size, hidden_size=args.decoder_hidden_size,
                      num_layers=args.decoder_num_layers)

    # optionally load pre-trained embeddings
    if args.embedding_type:
        encoder.embed.weight.data.copy_(field.vocab.vectors)
        decoder.embed.weight.data.copy_(field.vocab.vectors)

    # whether we will propagate gradients to word embeddings
    encoder.embed.weight.require_grads = args.train_embeddings
    decoder.embed.weight.require_grads = args.train_embeddings

    seq2seq = cuda(Seq2Seq(encoder, decoder, vocab_size))
    print(seq2seq)  # print model summary

    optimizer = optim.Adam(seq2seq.parameters(), lr=args.learning_rate)

    best_val_loss = None
    for epoch in range(args.max_epochs):
        # calculate train and val loss
        train_loss = train(seq2seq, optimizer, train_iter, vocab_size, args.gradient_clip, padding_idx)
        val_loss = evaluate(seq2seq, val_iter, vocab_size, padding_idx)
        print("[Epoch=%d] train_loss %f - val_loss %f" % (epoch, train_loss, val_loss))

        # save model if model achieved best val loss
        if not best_val_loss or val_loss < best_val_loss:
            print('Saving model...')
            save_model(seq2seq, epoch, val_loss, field)
            best_val_loss = val_loss


if __name__ == '__main__':
    main()
