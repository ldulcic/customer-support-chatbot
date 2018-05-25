import torch
import torch.nn as nn
import random


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

        hidden = h_n[:self.decoder.num_layers]  # output of all encoder layers for t=seq_len
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
