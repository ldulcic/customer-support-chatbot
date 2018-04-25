import torch
import dill
from model.model import Encoder, Decoder, Seq2Seq
from constants import SOS_TOKEN, EOS_TOKEN
from util import cuda


def prepare_question(question, field):
    arr = [field.preprocess(question.lower())]
    return field.numericalize(arr, device=-1)


def main():
    model_path = '.save/seq2seq-0-6.720723.pt'
    field_path = '.save/field'

    field = dill.load(open(field_path, 'rb'))

    embedding_size = 20
    hidden_size = 100
    vocab_size = len(field.vocab)

    encoder = Encoder(vocab_size, embedding_size, hidden_size)
    decoder = Decoder(vocab_size, embedding_size, hidden_size)
    seq2seq = cuda(Seq2Seq(encoder, decoder, vocab_size))
    seq2seq.load_state_dict(torch.load(model_path))
    seq2seq.eval()

    question = ''
    print('Hi, how can I help you?')
    while question != 'bye':
        question = input()

        tensor = prepare_question(question, field)

        token_idx = seq2seq.predict(tensor, field.vocab.stoi[SOS_TOKEN], field.vocab.stoi[EOS_TOKEN])
        response = ' '.join(map(lambda idx: field.vocab.itos[idx], token_idx))
        print(response)


if __name__ == '__main__':
    main()
