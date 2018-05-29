import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod

# TODO comments, implementation

init_map = {
    'zeros': lambda args: ZerosInit(args.decoder_num_layers, args.decoder_hidden_size),
    'bahdanau': lambda args: BahdanauInit(args.encoder_hidden_size, args.decoder_hidden_size),
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

    def __init__(self, decoder_num_layers, decoder_hidden_size):
        self.decoder_num_layers = decoder_num_layers
        self.decoder_hidden_size = decoder_hidden_size
        super(ZerosInit, self).__init__()

    def forward(self, h_n):
        batch_size = h_n.size(1)
        return torch.zeros(self.decoder_num_layers, batch_size, self.decoder_hidden_size)


class BahdanauInit(DecoderInit):
    def __init__(self, encoder_hidden_size, decoder_hidden_size):
        super(BahdanauInit, self).__init__()
        self.linear = nn.Linear(in_features=encoder_hidden_size, out_features=decoder_hidden_size)

    def forward(self, h_n):
        num_hidden_states = h_n.size(0)
        backward_h = h_n[torch.range(1, num_hidden_states, 2).long()]  # take backward encoder RNN states
        return F.tanh(self.linear(backward_h))
