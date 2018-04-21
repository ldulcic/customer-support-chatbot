from torchtext import data
from constants import SOS_TOKEN, EOS_TOKEN, CUDA
import preprocessor as p


def tokenize(sent):
    print(p.tokenize(p.clean(sent)))
    return p.tokenize(p.clean(sent))


def load_dataset():
    field = data.Field(init_token=SOS_TOKEN, eos_token=EOS_TOKEN,
                       tokenize='spacy', lower=True)

    # load dataset
    train, val, test = data.TabularDataset.splits(
        path='data/twitter_customer_support',
        format='tsv',
        train='twitter_customer_support-small-train.tsv',
        validation='twitter_customer_support-small-val.tsv',
        test='twitter_customer_support-small-test.tsv',
        fields=[
            ('question', field),
            ('answer', field)
        ]
    )

    # build vocabulary
    field.build_vocab(train, vectors='glove.twitter.27B.25d')

    # create iterators for dataset
    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train, val, test), batch_size=32, sort_key=lambda x: max(len(x.question), len(x.answer)),
        device=0 if CUDA else -1, repeat=False)

    return field.vocab, train_iter, val_iter, test_iter


if __name__ == '__main__':
    load_dataset()
