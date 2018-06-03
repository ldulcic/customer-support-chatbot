import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod


class SequenceSampler(ABC):
    """
    Samples output sequence from decoder given input sequence (encoded). Sequence will be sampled until EOS token is
    sampled or sequence reaches ``max_length``.

    Inputs: encoder_outputs, h_n, decoder, sos_idx, eos_idx, max_length
        - **encoder_outputs** (seq_len, batch, encoder_hidden_size): Last encoder layer outputs for every timestamp.
        - **h_n** (num_layers * num_directions, batch, hidden_size): RNN outputs for all layers for t=seq_len (last
                    timestamp)
        - **decoder**: Seq2seq decoder.
        - **sos_idx** (scalar): Index of start of sentence token in vocabulary.
        - **eos_idx** (scalar):Index of end of sentence token in vocabulary.
        - **max_length** (scalar): Maximum length of sampled sequence.

    Outputs: sequences, lengths
        - **sequences** (max_seq_len, batch): Sampled sequences.
        - **lengths** (batch): Length of sequence for every batch.
    """

    @abstractmethod
    def sample(self, encoder_outputs, h_n, decoder, sos_idx, eos_idx, max_length):
        raise NotImplementedError


class GreedySampler(SequenceSampler):
    """
    Greedy sampler always chooses the most probable next token when sampling sequence.

    Inputs: encoder_outputs, h_n, decoder, sos_idx, eos_idx, max_length
        - **encoder_outputs** (seq_len, batch, encoder_hidden_size): Last encoder layer outputs for every timestamp.
        - **h_n** (num_layers * num_directions, batch, hidden_size): RNN outputs for all layers for t=seq_len (last
                    timestamp)
        - **decoder**: Seq2seq decoder.
        - **sos_idx** (scalar): Index of start of sentence token in vocabulary.
        - **eos_idx** (scalar):Index of end of sentence token in vocabulary.
        - **max_length** (scalar): Maximum length of sampled sequence.

    Outputs: sequences, lengths
        - **sequences** (batch, max_seq_len): Sampled sequences.
        - **lengths** (batch): Length of sequence for every batch.
    """
    def sample(self, encoder_outputs, h_n, decoder, sos_idx, eos_idx, max_length):
        batch_size = encoder_outputs.size(1)
        sequences = None

        input_word = torch.tensor([sos_idx] * batch_size)
        kwargs = {}
        for t in range(max_length):
            output, attn_weights, kwargs = decoder(t, input_word, encoder_outputs, h_n, **kwargs)
            _, argmax = output.max(dim=1)  # greedily take the most probable word
            input_word = argmax
            argmax = argmax.unsqueeze(1)  # (batch) -> (batch, 1) because of concatenating to sequences
            sequences = argmax if sequences is None else torch.cat([sequences, argmax], dim=1)

        # ensure there is EOS token at the end of every sequence (important for calculating lengths)
        end = torch.tensor([eos_idx] * batch_size).unsqueeze(1)  # (batch, 1)
        sequences = torch.cat([sequences, end], dim=1)

        # calculate lengths
        _, lengths = (sequences == eos_idx).max(dim=1)

        return sequences, lengths


class RandomSampler(SequenceSampler):
    """
    Random sampler uses roulette-wheel when selecting next token in sequence, tokens (softmax) probabilities are used as
    token weights in roulette-wheel.

    Inputs: encoder_outputs, h_n, decoder, sos_idx, eos_idx, max_length
        - **encoder_outputs** (seq_len, batch, encoder_hidden_size): Last encoder layer outputs for every timestamp.
        - **h_n** (num_layers * num_directions, batch, hidden_size): RNN outputs for all layers for t=seq_len (last
                    timestamp)
        - **decoder**: Seq2seq decoder.
        - **sos_idx** (scalar): Index of start of sentence token in vocabulary.
        - **eos_idx** (scalar):Index of end of sentence token in vocabulary.
        - **max_length** (scalar): Maximum length of sampled sequence.

    Outputs: sequences, lengths
        - **sequences** (batch, max_seq_len): Sampled sequences.
        - **lengths** (batch): Length of sequence for every batch.
    """
    def sample(self, encoder_outputs, h_n, decoder, sos_idx, eos_idx, max_length):
        batch_size = encoder_outputs.size(1)
        sequences = None

        input_word = torch.tensor([sos_idx] * batch_size)
        kwargs = {}
        for t in range(max_length):
            output, attn_weights, kwargs = decoder(t, input_word, encoder_outputs, h_n, **kwargs)
            indices = torch.multinomial(F.softmax(output, dim=1), 1)  # roulette-wheel selection of tokens with probability as weights (batch, 1)
            input_word = indices.squeeze(1)  # (batch, 1) -> (batch)
            sequences = indices if sequences is None else torch.cat([sequences, indices], dim=1)

        # ensure there is EOS token at the end of every sequence (important for calculating lengths)
        end = torch.tensor([eos_idx] * batch_size).unsqueeze(1)  # (batch, 1)
        sequences = torch.cat([sequences, end], dim=1)

        # calculate lengths
        _, lengths = (sequences == eos_idx).max(dim=1)

        return sequences, lengths


class BeamSearch(SequenceSampler):
    # TODO
    def sample(self, encoder_outputs, h_n, decoder, sos_idx, eos_idx, max_length):
        raise NotImplementedError
