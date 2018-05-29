import torch.nn as nn
from util import RNNWrapper
from abc import ABC, abstractmethod


def encoder_factory(args, vocab_size, padding_idx):
    return SimpleEncoder(
        rnn_cls=getattr(nn, args.encoder_rnn_cell),  # gets LSTM or GRU constructor from nn module
        vocab_size=vocab_size,
        embed_size=args.embedding_size,
        hidden_size=args.encoder_hidden_size,
        padding_idx=padding_idx,
        num_layers=args.encoder_num_layers,
        dropout=args.encoder_rnn_dropout,
        bidirectional=args.encoder_bidirectional
    )


class Encoder(ABC, nn.Module):
    """
    Defines encoder for seq2seq model.

    Inputs: input, h_0
        - **input** (seq_length, batch_size): Input sequence.
        - **h_0** (num_layers * num_directions, batch, hidden_size): Initial hidden state of RNN. Default: None.

    Outputs: outputs, h_n
        - **outputs** (seq_len, batch, hidden_size * num_directions): Outputs of RNN last layer for every timestamp.
        - **h_n** (num_layers * num_directions, batch, hidden_size): RNN outputs for all layers for t=seq_len (last
                    timestamp)
    """

    @abstractmethod
    def forward(self, input, h_0=None):
        pass

    @property
    @abstractmethod
    def hidden_size(self):
        pass

    @property
    @abstractmethod
    def bidirectional(self):
        pass

    @property
    @abstractmethod
    def num_layers(self):
        pass


class SimpleEncoder(Encoder):
    """
    Encoder for seq2seq models.

    :param rnn_cls: RNN callable constructor. RNN is either LSTM or GRU.
    :param vocab_size: Size of vocabulary over which we operate.
    :param embed_size: Dimensionality of word embeddings.
    :param hidden_size: Dimensionality of RNN hidden representation.
    :param num_layers: Number of layers in RNN. Default: 1.
    :param dropout: Dropout probability for RNN. Default: 0.2.
    :param bidirectional: If True, RNN will be bidirectional. Default: False.

    Inputs: input, h_0
        - **input** (seq_length, batch_size): Input sequence.
        - **h_0** (num_layers * num_directions, batch, hidden_size): Initial hidden state of RNN. Default: None.

    Outputs: outputs, h_n
        - **outputs** (seq_len, batch, hidden_size * num_directions): Outputs of RNN last layer for every timestamp.
        - **h_n** (num_layers * num_directions, batch, hidden_size): RNN outputs for all layers for t=seq_len (last
                    timestamp)
    """

    def __init__(self, rnn_cls, vocab_size, embed_size, hidden_size, padding_idx, num_layers=1, dropout=0.2,
                 bidirectional=False):
        super(SimpleEncoder, self).__init__()

        self._hidden_size = hidden_size
        self._bidirectional = bidirectional
        self._num_layers = num_layers

        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size, padding_idx=padding_idx)
        self.rnn = RNNWrapper(rnn_cls(input_size=embed_size,
                                      hidden_size=hidden_size,
                                      num_layers=num_layers,
                                      dropout=dropout,
                                      bidirectional=bidirectional))

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def bidirectional(self):
        return self._bidirectional

    @property
    def num_layers(self):
        return self._num_layers

    def forward(self, input, h_0=None):
        embedded = self.embed(input)
        outputs, h_n = self.rnn(embedded, h_0)
        return outputs, h_n
