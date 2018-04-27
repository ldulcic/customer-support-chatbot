import os
import spacy
from torchtext import data
from sklearn.model_selection import train_test_split
#from torchtext.datasets.translation import TranslationDataset
from constants import SOS_TOKEN, EOS_TOKEN, CUDA
import preprocessor as prepro


# class TwitterCustomerSupportDataset(TranslationDataset):
#     urls = ['https://www.kaggle.com/thoughtvector/customer-support-on-twitter/downloads/twcs.zip/10']
#     name = 'twitter-customer-support'
#     dirname = ''


def split_dataset(path, random_state=287):
    dir_name = os.path.dirname(path)
    with open(path, encoding='utf-8') as fd:
        lines = fd.readlines()

    train, rest = train_test_split(lines, test_size=0.4, random_state=random_state)
    val, test = train_test_split(rest, test_size=0.5, random_state=random_state)

    # write train
    with open(dir_name + os.path.sep + 'twitter_customer_support-train.tsv', 'w', encoding='utf-8') as fd:
        fd.writelines(train)

    # write val
    with open(dir_name + os.path.sep + 'twitter_customer_support-val.tsv', 'w', encoding='utf-8') as fd:
        fd.writelines(val)

    # write test
    with open(dir_name + os.path.sep + 'twitter_customer_support-test.tsv', 'w', encoding='utf-8') as fd:
        fd.writelines(test)


def load_dataset(args):
    spacy_en = spacy.load('en')

    def tokenize(text):
        tok = [token.text for token in spacy_en.tokenizer(prepro.clean(text))]
        # if len(tok) == 0:
        #     print('Empty!')
        return tok

    def m(e):
        return hasattr(e, 'answer') and hasattr(e, 'question') and e.answer and e.question

    field = data.Field(init_token=SOS_TOKEN, eos_token=EOS_TOKEN,
                       tokenize=tokenize, lower=True)

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
        ],
        filter_pred=m  # remove examples with empty question or empty answer
    )

    # build vocabulary
    field.build_vocab(train, vectors=args.embedding_type, max_size=20000, min_freq=2)

    # create iterators for dataset
    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train, val, test), batch_size=args.batch_size, sort_key=lambda x: max(len(x.question), len(x.answer)), # TODO should it be max (len, len) ?
        device=0 if CUDA else -1, repeat=False)

    return field, train_iter, val_iter, test_iter


if __name__ == '__main__':
    split_dataset('../data/twitter_customer_support/twitter_customer_support.tsv')
