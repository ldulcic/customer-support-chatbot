import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from .embeddings import embeddings_factory
from .attention import attention_factory
from .decoder_init import decoder_init_factory


def bahdanau_decoder_factory(args, embed, attn, init, metadata):
    return BahdanauDecoder(
        rnn_cls=getattr(nn, args.encoder_rnn_cell),  # gets LSTM or GRU constructor from nn module
        embed=embed,
        attn=attn,
        init_hidden=init,
        vocab_size=metadata.vocab_size,
        embed_size=args.embedding_size,
        rnn_hidden_size=args.decoder_hidden_size,
        encoder_hidden_size=args.encoder_hidden_size * (2 if args.encoder_bidirectional else 1),
        num_layers=args.decoder_num_layers,
        dropout=args.decoder_rnn_dropout
    )


def luong_decoder_factory(args, embed, attn, init, metadata):
    return LuongDecoder(
        rnn_cls=getattr(nn, args.encoder_rnn_cell),  # gets LSTM or GRU constructor from nn module
        embed=embed,
        attn=attn,
        init_hidden=init,
        vocab_size=metadata.vocab_size,
        embed_size=args.embedding_size,
        rnn_hidden_size=args.decoder_hidden_size,
        attn_hidden_projection_size=args.luong_attn_hidden_size,
        encoder_hidden_size=args.encoder_hidden_size * (2 if args.encoder_bidirectional else 1),
        num_layers=args.decoder_num_layers,
        dropout=args.decoder_rnn_dropout,
        input_feed=args.luong_input_feed
    )


decoder_map = {
    'bahdanau': bahdanau_decoder_factory,
    'luong': luong_decoder_factory
}


def decoder_factory(args, metadata):
    """
    Returns instance of ``Decoder`` based on provided args.
    """
    # TODO what if attention type is 'none' ?
    embed = embeddings_factory(args, metadata)
    attn = attention_factory(args)
    init = decoder_init_factory(args)
    return decoder_map[args.decoder_type](args, embed, attn, init, metadata)


class Decoder(ABC, nn.Module):
    """
    Defines decoder for seq2seq models. Decoder is designed to be iteratively called until caller decides to stop.
    In every step decoder depends on output in previous timestamps and encoder outputs.

    Decoder is designed to be stateless, to achieve that decoder doesn't save any data which will be required to
    calculate output in the next timestamp, but instead, it returns that data to caller and is expecting that caller
    forwards that same data in next timestamp (if caller wants to run another timestamp). That's why decoder accepts
    and return `kwargs` in every timestamp. Caller should not alter or touch `kwargs` in any way, `kwargs` should
    just be taken as return value and forwarded in the next timestamp. If caller alters `kwargs` decoder behaviour is
    undefined and will most probably malfunction in that case. It is very important that caller handles decoder in
    previously explained steps. When calling decoder for the first time (t=0) just forward empty dictionary as `kwargs`.

   Following code shows how to use instance of decoder::

        >>> kwargs = {}
        >>> for t in range(num_timestamps):
        >>>    output, attn_weights, kwargs = decoder(t, input_word, encoder_outputs, h_n, **kwargs)
        >>>    # your code (remember - don't touch kwargs)

    Inputs: t, input, encoder_outputs, h_n, kwargs
        - **t** (scalar): Current timestamp in decoder (0-based).
        - **input** (batch): Input word.
        - **encoder_outputs** (seq_len, batch, encoder_hidden_size): Last encoder layer outputs for every timestamp.
        - **h_n** (num_layers * num_directions, batch, hidden_size): RNN outputs for all layers for t=seq_len (last
                    timestamp)
        - **kwargs** Dictionary of additional args used by decoder.

    Outputs: output, attn_weights, kwargs
        - **output** (batch, vocab_size): (Raw unscaled logits) Predictions for next word in output sequence.
        - **attn_weights** (batch, seq_len): (Optional) Attention weights. This value is returned only if decoder uses
        attention.
        - **kwargs** Dictionary of additional args used by decoder.
    """

    def __init__(self, *args):
        super(Decoder, self).__init__()
        self._args = []
        self._args_init = {}

    def forward(self, t, input, encoder_outputs, h_n, **kwargs):
        """
        This method is decorator around ``_forward`` which handles additional args (kwargs).

        This method will, based on subclass configured ``args``, unpack ``kwargs`` and forward it to subclass
        implemented ``_forward`` method. It will also unpack ``_forward`` additional args and pack them in ``kwargs``
        to return them to caller. It is very important that subclass implementation of ``_forward`` returns additional
        args in the SAME ORDER as it receives them. ``_forward`` will receive additional args in the order it puts them
        in ``args`` list.

        Additionally, this method will in first timestamp (t=0) init additional args with subclass provided methods
        in ``args_init``. (This also means that any data provided in kwargs when t=0 will be ignored, kwargs should be
        empty dictionary when t=0)
        """
        assert (t == 0 and not kwargs) or (t > 0 and kwargs)

        extra_args = []
        for arg in self.args:
            if t > 0 and arg not in kwargs:
                raise AttributeError("Mandatory arg \"%s\" not present among method arguments" % arg)
            extra_args.append(self.args_init[arg](encoder_outputs, h_n) if t == 0 else kwargs[arg])

        output, attn_weights, *out = self._forward(t, input, encoder_outputs, *extra_args)
        return output, attn_weights, {k: v for k, v in zip(self.args, out)}

    @abstractmethod
    def _forward(self, *args):
        """
        Implements decoder forward pass. Subclasses should list args that want to receive in this method. Mandatory
        ones are ``t, input, encoder_outputs, h_n`` plus any number of additional ones. Additional arguments will be
        forwarded to this method in the SAME ORDER they are listed in ``args`` property. Also additional args should be
        returned in the SAME ORDER as received.

        Example of how should ``_forward`` be implemented (pay attention to ordering of additional args)::

            >>> args = ['arg1', 'arg2']
            >>> def _forward(t, input, encoder_outputs, h_n, arg1, arg2):
            >>>     ...
            >>>     arg1_new = some_calculation1()
            >>>     arg2_new = some_calculation2()
            >>>     ...
            >>>     return output, attn_weights, arg1_new, arg2_new
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def hidden_size(self):
        raise AttributeError

    @property
    @abstractmethod
    def num_layers(self):
        raise AttributeError

    @property
    @abstractmethod
    def has_attention(self):
        raise AttributeError

    @property
    def args(self):
        """
        List of additional arguments concrete subclass wants to receive.
        """
        return self._args

    @args.setter
    def args(self, value):
        self._args = value

    @property
    def args_init(self):
        """
        Dictionary which provides for every additional argument provides function which returns initial value for that
        parameter. Dictionary keys are additional args names, ``args_init`` must contain entry for every arg from
        ``args``. This dictionary is used in the first timestamp (t=0) to provide initial values for additional args (
        see ``forward``). Init functions will receive ``encoder_outputs, h_n`` as input.
        """
        return self._args_init

    @args_init.setter
    def args_init(self, value):
        self._args_init = value


class BahdanauDecoder(Decoder):
    """
    Bahdanau decoder for seq2seq models. This decoder is called Bahdanau because it's implemented similar to decoder
    from (Bahdanau et al., 2015.) paper (see paper for details of inner workings of model). Decoder doesn't need to work
    exactly like Bahdanau paper decoder because a lot of things can be configured, like decoder RNN initialization,
    attention type and RNN cell type.

    If you want this decoder to work exactly like Bahdanau, use ``global attention`` with ``concat`` score function,
    use `bahdanau` decoder init function, use ``GRU`` RNN cell and make your encoder bi-directional.

    :param rnn_cls: RNN callable constructor. RNN is either LSTM or GRU.
    :param embed: Embedding layer.
    :param attn: Attention layer.
    :param init_hidden: Function for generating initial RNN hidden state.
    :param vocab_size: Size of vocabulary over which we operate.
    :param embed_size: Dimensionality of word embeddings.
    :param rnn_hidden_size: Dimensionality of RNN hidden representation.
    :param encoder_hidden_size: Dimensionality of encoder hidden representation (important for calculating attention
                                context)
    :param num_layers: Number of layers in RNN. Default: 1.
    :param dropout: RNN dropout layers mask probability. Default: 0.2.

    Inputs: t, input, encoder_outputs, h_n, last_hidden
        - **t** (scalar): Current timestamp in decoder (0-based).
        - **input** (batch): Input word.
        - **encoder_outputs** (seq_len, batch, encoder_hidden_size): Last encoder layer outputs for every timestamp.
        - **h_n** (num_layers * num_directions, batch, hidden_size): RNN outputs for all layers for t=seq_len (last
                    timestamp)
        - **last_state** (num_layers, batch, hidden_size): (Additional arg) RNN state from previous timestamp. This
        state is a hidden state (tensor) if rnn cell type is GRU, if rnn cell type is LSTM this is a tuple of last
        hidden state and last cell state.

    Outputs: output, attn_weights, hidden
        - **output** (batch, vocab_size): (Raw unscaled logits) Predictions for next word in output sequence.
        - **attn_weights** (batch, seq_len): (Optional) Attention weights. This value is returned only if decoder uses
        attention.
        - **hidden** (num_layers, batch, hidden_size): (Additional arg) New RNN state. (hidden state if rnn is GRU,
        tuple of hidden and cell state if rnn is LSTM)
    """

    LAST_STATE = 'last_state'

    args = [LAST_STATE]

    def __init__(self, rnn_cls, embed, attn, init_hidden, vocab_size, embed_size, rnn_hidden_size, encoder_hidden_size,
                 num_layers=1, dropout=0.2):
        super(BahdanauDecoder, self).__init__()

        self.args_init = {
            self.LAST_STATE: lambda encoder_outputs, h_n: self.initial_hidden(h_n)
        }

        if rnn_hidden_size % 2 != 0:
            raise ValueError('RNN hidden size must be divisible by 2 because of maxout layer.')

        self._hidden_size = rnn_hidden_size
        self._num_layers = num_layers

        self.initial_hidden = init_hidden
        self.embed = embed
        self.rnn = rnn_cls(input_size=embed_size + encoder_hidden_size,
                           hidden_size=rnn_hidden_size,
                           num_layers=num_layers,
                           dropout=dropout)
        self.attn = attn
        self.out = nn.Linear(in_features=rnn_hidden_size // 2, out_features=vocab_size)

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def num_layers(self):
        return self._num_layers

    @property
    def has_attention(self):
        return True

    def _forward(self, t, input, encoder_outputs, last_state):
        embedded = self.embed(input)

        # if RNN is LSTM state is tuple
        last_hidden = last_state[0] if isinstance(last_state, tuple) else last_state

        # prepare rnn input
        attn_weights, context = self.attn(t, last_hidden[-1], encoder_outputs)
        rnn_input = torch.cat([embedded, context], dim=1)
        rnn_input = rnn_input.unsqueeze(0)  # (batch, embed + enc_h) -> (1, batch, embed + enc_h)

        # calculate rnn output
        _, state = self.rnn(rnn_input, last_state)

        # if RNN is LSTM state is tuple
        hidden = state[0] if isinstance(state, tuple) else state

        # maxout layer (with k=2)
        top_layer_hidden = hidden[-1]  # (batch, rnn_hidden)
        batch_size = top_layer_hidden.size(0)
        maxout_input = hidden[-1].view(batch_size, -1, 2)  # (batch, rnn_hidden) -> (batch, rnn_hidden/2, 2) k=2
        maxout, _ = maxout_input.max(dim=2)  # (batch, rnn_hidden/2)

        # calculate logits
        output = self.out(maxout)

        return output, attn_weights, state


class LuongDecoder(Decoder):
    """
    Luong decoder for seq2seq models. This decoder is called Luong because it's implemented like decoder
    from (Luong et al., 2015.) paper (see paper for details of inner workings of model). This decoder implementation
    supports all variations of Luong decoder, plus you can try out this decoder with GRU RNN cell (Luong used only LSTM)

    :param rnn_cls: RNN callable constructor. RNN is either LSTM or GRU.
    :param embed: Embedding layer.
    :param attn: Attention layer.
    :param init_hidden: Function for generating initial RNN hidden state.
    :param vocab_size: Size of vocabulary over which we operate.
    :param embed_size: Dimensionality of word embeddings.
    :param rnn_hidden_size: Dimensionality of RNN hidden representation.
    :param attn_hidden_projection_size: Dimensionality of hidden state produced by combining RNN hidden state and
                                attention context. h_att = tanh( W * [c;h_rnn] )
    :param encoder_hidden_size: Dimensionality of encoder hidden representation (important for calculating attention
                                context)
    :param num_layers: Number of layers in RNN. Default: 1.
    :param dropout: RNN dropout layers mask probability. Default: 0.2.
    :param input_feed: If True input feeding approach will be used. Input feeding approach feeds previous attentional
    hidden state to RNN in current timestamp (so decoder can be aware of previous alignment decisions). Default: False.

    Inputs: t, input, encoder_outputs, h_n, last_hidden, last_attn_hidden
        - **t** (scalar): Current timestamp in decoder (0-based).
        - **input** (batch): Input word.
        - **encoder_outputs** (seq_len, batch, encoder_hidden_size): Last encoder layer outputs for every timestamp.
        - **h_n** (num_layers * num_directions, batch, hidden_size): RNN outputs for all layers for t=seq_len (last
                    timestamp)
        - **last_state** (num_layers, batch, hidden_size): (Additional arg) RNN state from previous timestamp. This
        state is a hidden state (tensor) if rnn cell type is GRU, if rnn cell type is LSTM this is a tuple of last
        hidden state and last cell state.
        - **last_attn_hidden** (batch, attn_hidden): (Additional arg) Attention hidden state from previous timestamp.

    Outputs: output, hidden
        - **output** (batch, vocab_size): (Raw unscaled logits) Predictions for next word in output sequence.
        - **attn_weights** (batch, seq_len): (Optional) Attention weights. This value is returned only if decoder uses
        attention.
        - **hidden** (num_layers, batch, hidden_size): (Additional arg) New RNN state. (hidden state if rnn is GRU,
        tuple of hidden and cell state if rnn is LSTM)
        - **attn_hidden** (batch, attn_hidden): (Additional arg) New attention hidden state.
    """

    LAST_STATE = 'last_state'
    LAST_ATTN_HIDDEN = 'last_attn_hidden'

    args = [LAST_STATE]

    def __init__(self, rnn_cls, embed, attn, init_hidden, vocab_size, embed_size, rnn_hidden_size,
                 attn_hidden_projection_size, encoder_hidden_size, num_layers=1, dropout=0.2, input_feed=False):
        super(LuongDecoder, self).__init__()

        if input_feed:
            self.args += [self.LAST_ATTN_HIDDEN]

        self.args_init = {
            self.LAST_STATE: lambda encoder_outputs, h_n: self.initial_hidden(h_n),
            self.LAST_ATTN_HIDDEN: lambda encoder_outputs, h_n: self.last_attn_hidden_init(h_n.size(1))  # h_n.size(1) == batch_size
        }

        self._hidden_size = rnn_hidden_size
        self._num_layers = num_layers
        self.initial_hidden = init_hidden

        self.input_feed = input_feed
        self.attn_hidden_projection_size = attn_hidden_projection_size

        rnn_input_size = embed_size + (attn_hidden_projection_size if input_feed else 0)
        self.embed = embed
        self.rnn = rnn_cls(input_size=rnn_input_size,
                           hidden_size=rnn_hidden_size,
                           num_layers=num_layers,
                           dropout=dropout)
        self.attn = attn
        self.attn_hidden_lin = nn.Linear(in_features=rnn_hidden_size + encoder_hidden_size,
                                         out_features=attn_hidden_projection_size)
        self.out = nn.Linear(in_features=attn_hidden_projection_size, out_features=vocab_size)

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def num_layers(self):
        return self._num_layers

    @property
    def has_attention(self):
        return True

    def last_attn_hidden_init(self, batch_size):
        return torch.zeros(batch_size, self.attn_hidden_projection_size) if self.input_feed else None

    def _forward(self, t, input, encoder_outputs, last_state, last_attn_hidden=None):
        assert (self.input_feed and last_attn_hidden is not None) or (not self.input_feed and last_attn_hidden is None)

        embedded = self.embed(input)

        # prepare rnn input
        rnn_input = embedded
        if self.input_feed:
            rnn_input = torch.cat([rnn_input, last_attn_hidden], dim=1)
        rnn_input = rnn_input.unsqueeze(0)  # (batch, rnn_input_size) -> (1, batch, rnn_input_size)

        # rnn output
        _, state = self.rnn(rnn_input, last_state)

        # if RNN is LSTM state is tuple
        hidden = state[0] if isinstance(state, tuple) else state

        # attention context
        attn_weights, context = self.attn(t, hidden[-1], encoder_outputs)
        attn_hidden = torch.tanh(self.attn_hidden_lin(torch.cat([context, hidden[-1]], dim=1)))  # (batch, attn_hidden)

        # calculate logits
        output = self.out(attn_hidden)

        return output, attn_weights, state, attn_hidden
