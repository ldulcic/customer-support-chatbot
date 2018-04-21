import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm
from dataset import dataset_factory
from model.model import Encoder, Decoder, Seq2Seq
from util import cuda


def save_model(model, epoch, val_loss):
    if not os.path.isdir('.save'):
        os.makedirs('.save')
    torch.save(model.state_dict(), ".save/seq2seq-%d-%f.pt" % (epoch, val_loss))


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
        total_loss += loss.data[0]

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

        total_loss += loss.data[0]

        # clip gradients to avoid exploding gradient
        clip_grad_norm(model.parameters(), grad_clip)

        # update parameters
        optimizer.step()

    return total_loss / len(train_iter)


def main():
    vocab, train_iter, val_iter, test_iter = dataset_factory('twitter-customer-support')

    epochs = 100
    embedding_size = 20
    hidden_size = 100
    vocab_size = len(vocab)
    padding_idx = vocab.stoi['<pad>']

    encoder = Encoder(vocab_size, embedding_size, hidden_size)
    decoder = Decoder(vocab_size, embedding_size, hidden_size)
    seq2seq = cuda(Seq2Seq(encoder, decoder, vocab_size))

    optimizer = optim.Adam(seq2seq.parameters())

    best_val_loss = None
    for epoch in range(epochs):
        # calculate train and val loss
        train_loss = train(seq2seq, optimizer, train_iter, vocab_size, 5, padding_idx)
        val_loss = evaluate(seq2seq, val_iter, vocab_size, padding_idx)
        print("[Epoch=%d] train_loss %f - val_loss %f" % (epoch, train_loss, val_loss))

        # save model if model achieved best val loss
        if not best_val_loss or val_loss < best_val_loss:
            print('Saving model...')
            save_model(seq2seq, epoch, val_loss)
            best_val_loss = val_loss


if __name__ == '__main__':
    main()
