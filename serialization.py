import torch
import dill
import os


def ensure_dir_exists(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def save_field_and_config(dir_path, field, args):
    ensure_dir_exists(dir_path)
    with open(dir_path + '/field.dill', 'wb') as fd:
        dill.dump(field, fd)
    with open(dir_path + '/args.dill', 'wb') as fd:
        dill.dump(args, fd)


def save_model(dir_path, model, epoch, val_loss):
    ensure_dir_exists(dir_path)
    torch.save(model.state_dict(), "%s/seq2seq-%d-%f.pt" % (dir_path, epoch, val_loss))