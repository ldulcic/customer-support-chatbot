import torch
from models.seq2seq.encoder import encoder_factory
from models.seq2seq.decoder import decoder_factory
from models.seq2seq.model import Seq2SeqTrain, Seq2SeqPredict
from collections import OrderedDict


def train_model_factory(args, metadata):
    encoder = encoder_factory(args, metadata)
    decoder = decoder_factory(args, metadata)
    return Seq2SeqTrain(encoder, decoder, metadata.vocab_size)


def predict_model_factory(args, metadata, model_path, field):
    train_model = train_model_factory(args, metadata)
    train_model.load_state_dict(get_state_dict(args, model_path))
    return Seq2SeqPredict(train_model.encoder, train_model.decoder, field)


def get_state_dict(args, model_path):
    # load state dict and map it to current storage (CPU or GPU)
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)

    # if model was trained with DataParallel (on multiple GPUs) remove "module." at the beginning of every key in state
    # dict (so we can load model on 1 GPU or on CPU for inference)
    if args.cuda and args.multi_gpu:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            key = k[7:]  # remove "module."
            new_state_dict[key] = v
        return new_state_dict

    return state_dict
