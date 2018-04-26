import torch
import dill
import os


def ensure_dir_exists(path):
    if not os.path.isdir(path):
        os.makedirs(path)


# def save_field_and_args(dir_path, field, args):
#     ensure_dir_exists(dir_path)
#     with open(dir_path + '/field.dill', 'wb') as fd:
#         dill.dump(field, fd)
#     with open(dir_path + '/args.dill', 'wb') as fd:
#         dill.dump(args, fd)
#
#
# def load_field_and_args(dir_path):
#     with open(dir_path + '/field.dill', 'rb') as fd:
#         field = dill.load(fd)
#     with open(dir_path + '/args.dill', 'rb') as fd:
#         args = dill.load(fd)
#     return field, args


def save_object(obj, path):
    ensure_dir_exists(os.path.dirname(path))
    with open(path, 'wb') as fd:
        dill.dump(obj, fd)


def load_object(path):
    with open(path, 'rb') as fd:
        obj = dill.load(fd)
    return obj


def save_model(dir_path, model, epoch, val_loss):
    ensure_dir_exists(dir_path)
    torch.save(model.state_dict(), "%s/seq2seq-%d-%f.pt" % (dir_path, epoch, val_loss))
