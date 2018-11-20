import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from constants import LSTM, GRU

# TODO comments, implementation

init_map = {
    'zeros': lambda args: ZerosInit(args.decoder_num_layers, args.decoder_hidden_size, args.encoder_rnn_cell),
    'bahdanau': lambda args: BahdanauInit(args.encoder_hidden_size, args.decoder_num_layers, args.decoder_hidden_size,
                                          args.decoder_rnn_cell),
    'adjust_pad': None,  # TODO
    'adjust_all': None   # TODO
}


def decoder_init_factory(args):
    if args.decoder_init_type == 'bahdanau' and not args.encoder_bidirectional:
        raise AttributeError('Bahdanau decoder init requires encoder to be bidirectional.')
    return init_map[args.decoder_init_type](args)


class DecoderInit(ABC, nn.Module):
    @abstractmethod
    def forward(self, h_n):
        raise NotImplementedError


class ZerosInit(DecoderInit):

    def __init__(self, decoder_num_layers, decoder_hidden_size, rnn_cell_type):
        assert rnn_cell_type == LSTM or rnn_cell_type == GRU
        super(ZerosInit, self).__init__()
        self.decoder_num_layers = decoder_num_layers
        self.decoder_hidden_size = decoder_hidden_size
        self.rnn_cell_type = rnn_cell_type

    def forward(self, h_n):
        batch_size = h_n.size(1)
        hidden = torch.zeros(self.decoder_num_layers, batch_size, self.decoder_hidden_size)
        return hidden if self.rnn_cell_type == GRU else (hidden, hidden.clone())


class BahdanauInit(DecoderInit):
    def __init__(self, encoder_hidden_size, decoder_num_layers, decoder_hidden_size, rnn_cell_type):
        super(BahdanauInit, self).__init__()
        assert rnn_cell_type == LSTM or rnn_cell_type == GRU
        self.linear = nn.Linear(in_features=encoder_hidden_size, out_features=decoder_hidden_size)
        self.decoder_num_layers = decoder_num_layers
        self.decoder_hidden_size = decoder_hidden_size
        self.rnn_cell_type = rnn_cell_type

    def forward(self, h_n):
        num_hidden_states = h_n.size(0)
        batch_size = h_n.size(1)
        backward_h = h_n[torch.arange(1, num_hidden_states, 2)]  # take backward encoder RNN states
        hidden = torch.tanh(self.linear(backward_h))
        hidden = self.adjust_hidden_size(hidden)
        return hidden if self.rnn_cell_type == GRU else (hidden, torch.zeros(self.decoder_num_layers, batch_size,
                                                                             self.decoder_hidden_size))

    def adjust_hidden_size(self, hidden):
        """
        If encoder and decoder have different number of layers adjust size of initial hidden state for decoder
        by padding with zeros (when decoder has more layers) or slicing hidden state (when encoder has more layers)
        """
        num_layers = hidden.size(0)
        batch_size = hidden.size(1)
        hidden_size = hidden.size(2)

        if num_layers < self.decoder_num_layers:
            hidden = torch.cat([hidden, torch.zeros(self.decoder_num_layers - num_layers, batch_size, hidden_size)],
                               dim=0)

        if num_layers > self.decoder_num_layers:
            hidden = hidden[:self.decoder_num_layers]

        return hidden
