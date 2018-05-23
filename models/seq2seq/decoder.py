import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod


class Decoder(ABC, nn.Module):
    """
    Defines decoder for seq2seq models.

    Inputs: input, last_hidden, encoder_outputs
        - **t** (scalar): Current timestamp in decoder (0-based).
        - **input** (batch): Input word.
        - **last_hidden** (num_layers, batch, hidden_size): Last RNN hidden state, in first step
        this will be last hidden state of encoder and in subsequent steps it will be decoder hidden state from
        previous step.
        - **encoder_outputs** (seq_len, batch, encoder_hidden_size): Last encoder layer outputs for every timestamp.

    Outputs: output, hidden
        - **output** (batch, vocab_size): (Raw unscaled logits) Predictions for next word in output sequence.
        - **hidden** (num_layers, batch, hidden_size): New RNN hidden state.
        - **attn_weights** (batch, seq_len): (Optional) Attention weights. This value is returned only if decoder uses
        attention.
    """
    @abstractmethod
    def forward(self, t, input, last_hidden, encoder_outputs):
        pass

    @property
    @abstractmethod
    def hidden_size(self):
        pass

    @property
    @abstractmethod
    def num_layers(self):
        pass


class BahdanauDecoder(Decoder):
    """
    Bahdanau decoder for seq2seq models. This decoder is called Bahdanau because it works like decoder from (Bahdanau et
    al., 2014.) paper TODO give more details.

    :param attn: Attention layer.
    :param vocab_size: Size of vocabulary over which we operate.
    :param embed_size: Dimensionality of word embeddings.
    :param hidden_size: Dimensionality of RNN hidden representation.
    :param encoder_hidden_size: Dimensionality of encoder hidden representation (important for calculating attention
                                context)
    :param padding_idx: Index of pad token in vocabulary.
    :param num_layers: Number of layers in RNN. Default: 1.

    Inputs: input, last_hidden, encoder_outputs
        - **input** (batch): Input word.
        - **last_hidden** (num_layers, batch, hidden_size): Last RNN hidden state, in first step
        this will be last hidden state of encoder and in subsequent steps it will be decoder hidden state from
        previous step.
        - **encoder_outputs** (seq_len, batch, encoder_hidden_size): Last encoder layer outputs for every timestamp.

    Outputs: output, hidden
        - **output** (batch, vocab_size): (Raw unscaled logits) Predictions for next word in output sequence.
        - **hidden** (num_layers, batch, hidden_size): New RNN hidden state.
        - **attn_weights** (batch, seq_len): (Optional) Attention weights. This value is returned only if decoder uses
        attention.
    """
    def __init__(self, attn, vocab_size, embed_size, hidden_size, encoder_hidden_size, padding_idx, num_layers=1,
                 dropout=0.2):
        super(BahdanauDecoder, self).__init__()

        self._hidden_size = hidden_size
        self._num_layers = num_layers

        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size, padding_idx=padding_idx)
        self.rnn = nn.GRU(input_size=embed_size + encoder_hidden_size, hidden_size=hidden_size, num_layers=num_layers,
                          dropout=dropout)
        self.attn = attn
        # TODO this is not how Bahdanau calculates output
        self.out = nn.Linear(in_features=hidden_size, out_features=vocab_size)

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def num_layers(self):
        return self._num_layers

    def forward(self, t, input, last_hidden, encoder_outputs):
        embedded = self.embed(input)

        # prepare rnn input
        attn_weights, context = self.attn(t, last_hidden[-1], encoder_outputs)
        rnn_input = torch.cat([embedded, context], dim=1)
        rnn_input = rnn_input.unsqueeze(0)  # (batch, embed + enc_h) -> (1, batch, embed + enc_h)

        # calculate decoder output
        _, hidden = self.rnn(rnn_input, last_hidden)
        output = self.out(hidden[-1])  # hidden[-1] - hidden output of last layer

        return output, hidden, attn_weights
