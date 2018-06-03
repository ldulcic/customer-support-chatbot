import torch
from models.seq2seq.encoder import encoder_factory
from models.seq2seq.decoder import decoder_factory
from models.seq2seq.model import Seq2SeqTrain, Seq2SeqPredict


def train_model_factory(args, metadata):
    encoder = encoder_factory(args, metadata)
    decoder = decoder_factory(args, metadata)
    return Seq2SeqTrain(encoder, decoder, metadata.vocab_size)


def predict_model_factory(args, metadata, model_path, field):
    train_model = train_model_factory(args, metadata)
    train_model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    return Seq2SeqPredict(train_model.encoder, train_model.decoder, field)
