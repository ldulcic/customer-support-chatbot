import torch
import torch.nn as nn
import random
from constants import SOS_TOKEN, EOS_TOKEN
from .sampling import GreedySampler, RandomSampler, BeamSearch


class Seq2SeqTrain(nn.Module):
    def __init__(self, encoder, decoder, vocab_size, teacher_forcing_ratio=0.5):
        """
        Encapsulates Seq2Seq model. This model is used for training seq2seq model, it returns (unscaled)
        probabilities for output which is needed for model training.

        :param encoder: Encoder.
        :param decoder: Decoder.
        :param vocab_size: Vocabulary size.
        :param teacher_forcing_ratio: Teacher forcing ratio. Default: 0.5.

        Inputs: question, trg
            - **question** (seq_len + 2, batch): Question sequence. It is expected that sequences have <SOS> token at start
            and <EOS> token at the end. +2 because question contains two extra tokens <SOS> and <EOS>.
            - **answer** (seq_len + 2, batch): Answer sequence. It is expected that sequences have <SOS> token at start and
            <EOS> token at the end. +2 because answer contains two extra tokens <SOS> and <EOS>.

        Outputs: outputs
            - **outputs** (seq_len + 1, batch, vocab_size): Model predictions for output sequence. These are raw
            unscaled logits. First dimension is (seq_len + 1) because we return predictions for next word for all tokens
            except last one (<EOS>), which means seq2seq will feed in decoder following sequence [<SOS>, tok1, tok2, ...
            , tokN] (notice no <EOS> at the end) and return next word prediction for each one of them.
        """
        super(Seq2SeqTrain, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocab_size = vocab_size
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def forward(self, question, answer):
        answer_seq_len = answer.size(0)
        outputs = None

        # encode question sequence
        encoder_outputs, h_n = self.encoder(question)

        kwargs = {}
        input_word = answer[0]  # sos for whole batch
        for t in range(answer_seq_len - 1):
            output, attn_weights, kwargs = self.decoder(t, input_word, encoder_outputs, h_n, **kwargs)

            out = output.unsqueeze(0)  # (batch_size, vocab_size) -> (1, batch_size, vocab_size)
            outputs = out if outputs is None else torch.cat([outputs, out], dim=0)

            teacher_forcing = random.random() < self.teacher_forcing_ratio
            if teacher_forcing:
                input_word = answer[t + 1]  # +1 input word for next timestamp
            else:
                _, argmax = output.max(dim=1)
                input_word = argmax  # index of most probable token (for whole batch)

        return outputs


class Seq2SeqPredict(nn.Module):

    def __init__(self, encoder, decoder, sos_idx, eos_idx):
        super(Seq2SeqPredict, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.samplers = {
            'greedy': GreedySampler(),
            'random': RandomSampler(),
            'beam_search': BeamSearch()
        }

    def forward(self, question, sampling_strategy, max_seq_len):
        # encode question sequence
        encoder_outputs, h_n = self.encoder(question)
        # sample output sequence
        return self.samplers[sampling_strategy].sample(encoder_outputs, h_n, self.decoder, self.sos_idx, self.eos_idx,
                                                       max_seq_len)


class Seq2SeqPredictProxy(Seq2SeqPredict):

    def __init__(self, encoder, decoder, field):
        super().__init__(encoder, decoder, field.vocab.stoi[SOS_TOKEN], field.vocab.stoi[EOS_TOKEN])
        self.field = field

    def forward(self, question, sampling_strategy='greedy', max_seq_len=50):
        q = self.field.numericalize(question)
        sequences, lengths = super().forward(q, sampling_strategy, max_seq_len)

        batch_size = sequences.size(1)
        sequences, lengths = sequences.tolist(), lengths.tolist()

        seqs = []
        for batch in range(batch_size):
            seq = sequences[:lengths[batch], batch]
            seqs.append(' '.join(map(lambda tok: self.field.vocab.itos[tok], seq)))

        return seqs


