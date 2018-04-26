import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from util import cuda


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, bidirectional=False):
        """
        Encoder for seq2seq models.

        :param vocab_size: Size of vocabulary over which we operate.
        :param embed_size: Dimensionality of word embeddings.
        :param hidden_size: Dimensionality of RNN hidden representation.
        :param num_layers: Number of layers in RNN. Default: 1.
        :param bidirectional: If True, RNN will be bidirectional. Default: False.

        Inputs: input, h_0
            - **input** (seq_length, batch_size): Input sequence.
            - **h_0** (num_layers * num_directions, batch, hidden_size): Initial hidden state of RNN. Default: None.

        Outputs: outputs, h_n
            - **outputs** (seq_len, batch, hidden_size * num_directions): Outputs of RNN last layer for every timestamp.
            - **h_n** (num_layers * num_directions, batch, hidden_size): RNN outputs for all layers for t=seq_len (last timestamp)
        """
        super(Encoder, self).__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.rnn = nn.GRU(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers,
                          bidirectional=bidirectional)

    def forward(self, input, h_0=None):
        embedded = self.embed(input)
        outputs, h_n = self.rnn(embedded, h_0)
        return outputs, h_n


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1):
        """
        Decoder for seq2seq models.

        :param vocab_size: Size of vocabulary over which we operate.
        :param embed_size: Dimensionality of word embeddings.
        :param hidden_size: Dimensionality of RNN hidden representation.
        :param num_layers: Number of layers in RNN. Default: 1.

        Inputs: input, last_hidden
            - **input** (batch): Input word.
            - **last_hidden** (num_layers, batch, hidden_size): Last RNN hidden state, in first step
            this will be last hidden state of encoder and in subsequent steps it will be decoder hidden state from
            previous step.

        Outputs: output, hidden
            - **output** (batch, vocab_size): Prediction of next word in output sequence, logarithm of probability
            distribution over vocabulary (log_softmax).
            - **hidden** (num_layers, batch, hidden_size): New RNN hidden state.
        """
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.rnn = nn.GRU(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers)
        self.out = nn.Linear(in_features=hidden_size, out_features=vocab_size)

    def forward(self, input, last_hidden):
        embedded = self.embed(input).unsqueeze(0)  # (1,B,N)
        _, hidden = self.rnn(embedded, last_hidden)
        output = self.out(hidden[-1])  # hidden[-1] - hidden output of last layer
        output = F.log_softmax(output, dim=1)  # log softmax over logits to produce probability distributions over vocabulary
        return output, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, vocab_size):
        """
        Encapsulates Seq2Seq models. Currently doesn't support encoder and decoder with different number of layers.
        TODO support encoder and decoder with different number of layers.

        :param encoder: Encoder.
        :param decoder: Decoder.

        Inputs: src, trg
            - **src** (seq_len, batch): Source sequence.
            - **trg** (seq_len, batch): Target sequence. It is expected that sequences have <SOS> token at start and
            <EOS> token at the end.

        Outputs: outputs
            - **outputs** (seq_len, batch, vocab_size): Model predictions for output sequence.
        """
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocab_size = vocab_size

    def forward(self, src, trg):
        batch_size = src.size(1)
        trg_seq_len = trg.size(0) - 1  # - 1 because first token in every sequence is <sos> TODO note this in docs (dimensions don't match because we subtracted 1 from seq_len)
        outputs = cuda(Variable(torch.zeros(trg_seq_len, batch_size, self.vocab_size)))

        encoder_outputs, h_n = self.encoder(src)

        hidden = h_n  # output of all encoder layers for t=seq_len
        input_word = cuda(Variable(trg.data[0], requires_grad=False))  # sos for whole batch TODO check if we need to wrap tensor in new variable or just call trg[0] on existing variable, what's the difference?
        for t in range(trg_seq_len):
            output, hidden = self.decoder(input_word, hidden)
            outputs[t] = output
            max, argmax = output.data.max(dim=1)
            input_word = cuda(Variable(argmax))

        return outputs

    def predict(self, src, sos_idx, eos_idx):
        torch.no_grad()
        encoder_outputs, h_n = self.encoder(src)

        input_word = cuda(Variable(torch.LongTensor([sos_idx])))
        hidden = h_n

        out = []
        while len(out) < 10 or out[-1] == eos_idx:
            output, hidden = self.decoder(input_word, hidden)
            _, argmax = output.squeeze(0).data.max(dim=0)
            out.append(argmax.item())

        return out
