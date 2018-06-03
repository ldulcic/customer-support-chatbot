import torch
import pickle
import os
from constants import MODEL_FORMAT


def ensure_dir_exists(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def save_object(obj, path):
    ensure_dir_exists(os.path.dirname(path))
    with open(path, 'wb') as fd:
        pickle.dump(obj, fd)


def load_object(path):
    with open(path, 'rb') as fd:
        obj = pickle.load(fd)
    return obj


def save_vocab(vocab, path):
    """
    Saves Torchtext Field vocabulary. WARNING this method will erase vocab vectors!
    """
    # erasing vectors because we don't need them and they cause problems when loading model on CPU when model was
    # trained on GPU
    vocab.vectors = None
    save_object(vocab, path)


def save_model(dir_path, model, epoch, train_loss, val_loss):
    ensure_dir_exists(dir_path)
    torch.save(model.state_dict(), dir_path + os.path.sep + (MODEL_FORMAT % (epoch, train_loss, val_loss)))
