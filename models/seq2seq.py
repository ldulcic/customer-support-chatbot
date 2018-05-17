import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, padding_idx, num_layers=1, bidirectional=False):
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
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size, padding_idx=padding_idx)
        self.rnn = nn.GRU(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers,
                          bidirectional=bidirectional)

    def forward(self, input, h_0=None):
        embedded = self.embed(input)
        outputs, h_n = self.rnn(embedded, h_0)
        return outputs, h_n
    
    
class BahdanauAttnDecoder(nn.Module):
    """
    Bahdanau attention decoder for seq2seq models.

    :param vocab_size: Size of vocabulary over which we operate.
    :param embed_size: Dimensionality of word embeddings.
    :param hidden_size: Dimensionality of RNN hidden representation.
    :param encoder_hidden_size: Dimensionality of encoder hidden representation (important for calculating attention 
                                context)
    :param attn_hidden_size: Attention layer hidden size
    :param padding_idx: Index of pad token in vocabulary. 
    :param num_layers: Number of layers in RNN. Default: 1.

    Inputs: input, last_hidden
        - **input** (batch): Input word.
        - **last_hidden** (num_layers, batch, hidden_size): Last RNN hidden state, in first step
        this will be last hidden state of encoder and in subsequent steps it will be decoder hidden state from
        previous step.
        - **encoder_outputs** (seq_len, batch, encoder_hidden_size): Last encoder layer outputs for every timestamp.

    Outputs: output, hidden
        - **output** (batch, vocab_size): Prediction of next word in output sequence, logarithm of probability
        distribution over vocabulary (log_softmax).
        - **hidden** (num_layers, batch, hidden_size): New RNN hidden state.
    """
    def __init__(self, vocab_size, embed_size, hidden_size, encoder_hidden_size, attn_hidden_size, padding_idx, num_layers=1):
        super(BahdanauAttnDecoder, self).__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size, padding_idx=padding_idx)
        self.rnn = nn.GRU(input_size=embed_size + encoder_hidden_size, hidden_size=hidden_size, num_layers=num_layers)
        self.out = nn.Linear(in_features=hidden_size, out_features=vocab_size)

        # attention params
        self.W = nn.Linear(in_features=hidden_size + encoder_hidden_size, out_features=attn_hidden_size)
        self.v = nn.Linear(in_features=attn_hidden_size, out_features=1)

    def forward(self, input, last_hidden, encoder_outputs):
        embedded = self.embed(input)
        
        # prepare rnn input
        context = self.attn_context(last_hidden[-1], encoder_outputs)
        rnn_input = torch.cat([embedded, context], dim=1)
        rnn_input = rnn_input.unsqueeze(0)  # (batch, embed + enc_h) -> (1, batch, embed + enc_h)
        
        # calculate decoder output
        _, hidden = self.rnn(rnn_input, last_hidden)
        output = self.out(hidden[-1])  # hidden[-1] - hidden output of last layer
        
        return output, hidden

    def print_dim(self, name, tensor):
        print("%s -> %s" % (name, tensor.size()))
        
    def attn_context(self, hidden, encoder_outputs):
        """
        Generates attention context vector given last decoder hidden output and all encoder outputs.
        
        :param hidden (batch, hidden_size): Last decoder layer hidden output. 
        :param encoder_outputs (seq_len, batch, encoder_hidden_size): Last encoder layer outputs for every timestamp.
        :return: Attention context vector (batch, encoder_hidden_size)
        """
        energies = self.attn_energies(hidden, encoder_outputs)
        scores = F.softmax(energies, dim=0).unsqueeze(2) # (batch, seq_len, 1)
        enc_out = encoder_outputs.transpose(0, 1).transpose(1, 2)  # (seq_len, batch, enc_h) -> (batch, enc_h, seq_len)
        context = torch.bmm(enc_out, scores)  # (batch, enc_h, 1)
        return context.squeeze(2)

    def attn_energies(self, hidden, encoder_outputs):
        """
        Generates attention energies for current decoder position.

        :param hidden (batch, hidden_size): Last decoder layer hidden output.
        :param encoder_outputs (seq_len, batch, encoder_hidden_size): Last encoder layer outputs for every timestamp.
        :return: Attention energies (batch, seq_len)
        """
        seq_len = encoder_outputs.size(0)
        h = hidden.expand(seq_len, -1, -1)  # (batch, hidden_size) -> (seq_len, batch, hidden_size)
        energies = self.W(torch.cat([encoder_outputs, h], dim=2))  # (seq_len, batch, attn_h)
        energies = self.v(F.tanh(energies))  # (seq_len, batch, 1)
        return energies.squeeze(2).transpose(0, 1)


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, padding_idx, num_layers=1):
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
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size, padding_idx=padding_idx)
        self.rnn = nn.GRU(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers)
        self.out = nn.Linear(in_features=hidden_size, out_features=vocab_size)

    def forward(self, input, last_hidden):
        embedded = self.embed(input).unsqueeze(0)  # (1,B,N)
        _, hidden = self.rnn(embedded, last_hidden)
        output = self.out(hidden[-1])  # hidden[-1] - hidden output of last layer
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

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(1)
        trg_seq_len = trg.size(0) - 1  # - 1 because first token in every sequence is <sos> TODO note this in docs (dimensions don't match because we subtracted 1 from seq_len)
        outputs = torch.zeros(trg_seq_len, batch_size, self.vocab_size)

        encoder_outputs, h_n = self.encoder(src)

        hidden = h_n  # output of all encoder layers for t=seq_len
        input_word = trg[0]  # sos for whole batch
        for t in range(trg_seq_len):
            output, hidden = self.decoder(input_word, hidden, encoder_outputs)
            outputs[t] = output

            teacher_forcing = random.random() < teacher_forcing_ratio
            if teacher_forcing:
                input_word = trg[t + 1]  # +1 because trg contains <sos> at the beginning
            else:
                max, argmax = output.max(dim=1)
                input_word = argmax

        return outputs

    def predict(self, src, sos_idx, eos_idx):
        encoder_outputs, h_n = self.encoder(src)

        input_word = torch.tensor([sos_idx])
        hidden = h_n

        out = []
        while len(out) < 20:
            output, hidden = self.decoder(input_word, hidden)
            _, argmax = output.squeeze(0).data.max(dim=0)
            out.append(argmax.item())
            if out[-1] == eos_idx:
                break

        return out
