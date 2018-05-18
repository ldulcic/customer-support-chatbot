import torch
import os
import pandas as pd
from torchtext import data
from sklearn.model_selection import train_test_split
from constants import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN


def split_dataset(path, random_state=287):
    dir_name = os.path.dirname(path)
    file_name = os.path.basename(path).split('.')[0]

    df = pd.read_csv(path, sep='\t')

    train, rest = train_test_split(df, test_size=0.2, random_state=random_state)
    val, test = train_test_split(rest, test_size=0.5, random_state=random_state)

    # write train, val, test
    train.to_csv(dir_name + os.path.sep + file_name + '-train.tsv', sep='\t', index=False)
    val.to_csv(dir_name + os.path.sep + file_name + '-val.tsv', sep='\t', index=False)
    test.to_csv(dir_name + os.path.sep + file_name + '-test.tsv', sep='\t', index=False)


def load_dataset(args, device):
    split_dataset('data/twitter_customer_support/applesupport.tsv')

    field = data.Field(init_token=SOS_TOKEN, eos_token=EOS_TOKEN, pad_token=PAD_TOKEN,
                       tokenize='spacy', lower=True,
                       tensor_type=torch.cuda.LongTensor if device.type == 'cuda' else torch.LongTensor)

    # load dataset
    train, val, test = data.TabularDataset.splits(
        path='data/twitter_customer_support',
        format='tsv',
        train='applesupport-train.tsv',
        validation='applesupport-val.tsv',
        test='applesupport-test.tsv',
        fields=[
            ('question', field),
            ('answer', field)
        ],
        skip_header=True
    )

    # build vocabulary
    field.build_vocab(train, vectors=args.embedding_type, min_freq=2)

    # create iterators for dataset
    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train, val, test), batch_size=args.batch_size, sort_key=lambda x: len(x.question),
        device=device, repeat=False)

    return field, train_iter, val_iter, test_iter


if __name__ == '__main__':
    split_dataset('../../data/twitter_customer_support/applesupport.tsv')
