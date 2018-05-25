from models.seq2seq.encoder import encoder_factory
from models.seq2seq.decoder import decoder_factory
from models.seq2seq.model import Seq2Seq


def model_factory(args, field, vocab_size, padding_idx):
    encoder = encoder_factory(args, vocab_size, padding_idx)
    decoder = decoder_factory(args, vocab_size, padding_idx)

    # TODO refactor init of embeddings
    # optionally load pre-trained embeddings
    if args.embedding_type:
        encoder.embed.weight.data.copy_(field.vocab.vectors)
        decoder.embed.weight.data.copy_(field.vocab.vectors)

    # whether we will propagate gradients to word embeddings
    encoder.embed.weight.require_grads = args.train_embeddings
    decoder.embed.weight.require_grads = args.train_embeddings

    return Seq2Seq(encoder, decoder, vocab_size)
