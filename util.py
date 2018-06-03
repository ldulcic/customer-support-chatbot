import torch.nn as nn
import collections


def embedding_size_from_name(name):
    return int(name.strip().split('.')[-1][:-1])


def print_dim(name, tensor):
    print("%s -> %s" % (name, tensor.size()))


class RNNWrapper(nn.Module):
    """
    Wrapper around GRU or LSTM RNN. If underlying RNN is GRU, this wrapper does nothing, it just forwards inputs and
    outputs. If underlying RNN is LSTM this wrapper ignores LSTM cell state (s) and returns just hidden state (h).
    This wrapper allows us to unify interface for GRU and LSTM so we don't have to treat them differently.
    """

    LSTM = 'LSTM'
    GRU = 'GRU'

    def __init__(self, rnn):
        super(RNNWrapper, self).__init__()
        assert isinstance(rnn, nn.LSTM) or isinstance(rnn, nn.GRU)
        self.rnn_type = self.LSTM if isinstance(rnn, nn.LSTM) else self.GRU
        self.rnn = rnn

    def forward(self, *input):
        rnn_out, hidden = self.rnn(*input)
        if self.rnn_type == self.LSTM:
            hidden, s = hidden  # ignore LSTM cell state s
        return rnn_out, hidden


# Metadata used to describe dataset
Metadata = collections.namedtuple('Metadata', 'vocab_size padding_idx vectors')
