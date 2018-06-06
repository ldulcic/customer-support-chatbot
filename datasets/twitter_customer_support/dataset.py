import torch
from torchtext import data
from constants import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN
from util import Metadata


def load_metadata(vocab):
    """
    Loads dataset info data.
    """
    return Metadata(vocab_size=len(vocab), padding_idx=vocab.stoi[PAD_TOKEN], vectors=vocab.vectors)


def load_field(device):
    """
    Loads field for twitter dataset.
    """
    return data.Field(init_token=SOS_TOKEN, eos_token=EOS_TOKEN, pad_token=PAD_TOKEN,
                      tokenize='spacy', lower=True,
                      tensor_type=torch.cuda.LongTensor if device.type == 'cuda' else torch.LongTensor)


def load_dataset(args, device):
    field = load_field(device)

    # load dataset
    train, val, test = data.TabularDataset.splits(
        path='data/twitter_customer_support',
        format='tsv',
        train=args.dataset + '-train.tsv',
        validation=args.dataset + '-val.tsv',
        test=args.dataset + '-test.tsv',
        fields=[
            ('author_id', None),
            ('question', field),
            ('answer', field)
        ],
        skip_header=True
    )

    # build vocabulary
    field.build_vocab(train, vectors=args.embedding_type, min_freq=2, max_size=20000)

    # create iterators for dataset
    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train, val, test), batch_size=args.batch_size, sort_key=lambda x: len(x.question),
        device=device, repeat=False)

    vocab = field.vocab
    metadata = load_metadata(vocab)

    return metadata, field.vocab, train_iter, val_iter, test_iter
