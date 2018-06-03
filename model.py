import torch
from models.seq2seq.encoder import encoder_factory
from models.seq2seq.decoder import decoder_factory
from models.seq2seq.model import Seq2SeqTrain, Seq2SeqPredict
from constants import PAD_TOKEN


def train_model_factory(args, field, vocab_size):
    padding_idx = field.vocab.stoi[PAD_TOKEN]
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

    return Seq2SeqTrain(encoder, decoder, vocab_size)


def predict_model_factory(args, model_path, field, vocab_size):
    train_model = train_model_factory(args, field, vocab_size)
    train_model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    return Seq2SeqPredict(train_model.encoder, train_model.decoder, field)
