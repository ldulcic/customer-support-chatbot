import pandas as pd
import preprocessor as p
import os
import emoji
import re
import spacy
from sklearn.model_selection import train_test_split
from spacy_cld import LanguageDetector
from . import DATA_FOLDER


def id2text(df, x):
    row = df[df.tweet_id == int(x)]
    # some ids are missing, they just don't exist in data
    return '' if (len(row) == 0) else row.iloc[0]['text']


def clean_tweet(tweet):
    # removes @ mentions, hashtags, emojis, twitter reserved words and numbers
    p.set_options(p.OPT.EMOJI, p.OPT.MENTION, p.OPT.RESERVED, p.OPT.SMILEY, p.OPT.NUMBER)
    clean = p.clean(tweet)

    # transforms every url to "<url>" token and every hashtag to "<hashtag>" token
    p.set_options(p.OPT.EMOJI, p.OPT.MENTION, p.OPT.RESERVED, p.OPT.SMILEY, p.OPT.NUMBER, p.OPT.HASHTAG, p.OPT.URL)
    clean = p.tokenize(clean)
    clean = re.sub(r'\$HASHTAG\$', '<hashtag>', clean)
    clean = re.sub(r'\$URL\$', '<url>', clean)

    # preprocessor doesn't seem to clean all emojis so we run text trough emoji regex to clean leftovers
    clean = re.sub(emoji.get_emoji_regexp(), '', clean)

    # removing zero-width character which is often bundled with emojis
    clean = re.sub(u'\ufe0f', '', clean)

    # remove multiple empty spaces with one
    clean = re.sub(r' +', ' ', clean)

    # replace &gt; and &lt;
    clean = re.sub(r'&gt;', '>', clean)
    clean = re.sub(r'&lt;', '<', clean)

    # strip any leftover spaces at the beginning and end
    clean = clean.strip()

    return clean


def set_empty_if_not_english(nlp, x):
    doc = nlp(x)
    # TODO tweets with no language tend to be valid, but they also tend to be gibberish, filtering for now
    # TODO don't filter scotish, don't filter danish with "icloud" in it (for some reason every tweet with icloud in it is classified as danish), filtering for now
    return x if doc._.languages and ('en' in doc._.language_scores and doc._.language_scores['en'] >= 0.80) else ''


def qa_from_author(df, nlp, author_id):
    """
    Creates qa dataset (in form of dataframe) from all tweets of author (identified by author_id)

    :param df: All twitter customer support data as dataframe.
    :param author_id: Name of author.
    :return: Dataframe containing 'question' and 'answer' fields where 'question' is user tweet and 'answer' is customer
            support tweet
    """
    # get all tweets from certain support service
    support_service = df[df.author_id == author_id]
    # remove tweets which are not triggered by user tweet (there is no Q(uestion))
    support_service = support_service[~support_service.in_response_to_tweet_id.isnull()]

    # take column we are interested in
    support_service = support_service[['author_id', 'text', 'in_response_to_tweet_id']]

    # replace tweet ids with actual tweet text
    support_service.loc[:, 'in_response_to_tweet_id'] = support_service.in_response_to_tweet_id.apply(lambda x: id2text(df, x))

    # rename and rearrange columns
    support_service.rename(columns={'author_id': 'author_id', 'text': 'answer', 'in_response_to_tweet_id': 'question'},
                           inplace=True)
    support_service = support_service[['author_id', 'question', 'answer']]

    # clean twitter data
    support_service.loc[:, 'question'] = support_service.question.apply(clean_tweet)
    support_service.loc[:, 'answer'] = support_service.answer.apply(clean_tweet)

    # filter all languages which are not english (non-english tweets will be set to empty string and then filtered at
    # the end of this method)
    support_service.loc[:, 'question'] = support_service.question.apply(lambda x: set_empty_if_not_english(nlp, x))
    support_service.loc[:, 'answer'] = support_service.answer.apply(lambda x: set_empty_if_not_english(nlp, x))

    # remove all QA pairs where Q or A are empty or contain only dot (.)
    support_service = support_service[~(support_service.question == '') & ~(support_service.answer == '')]
    support_service = support_service[~(support_service.question == '.') & ~(support_service.answer == '.')]

    return support_service


def split_dataset(path, random_state=287):
    dir_name = os.path.dirname(path)
    file_name = os.path.basename(path).split('.')[0]  # file name must end in '.tsv'

    df = pd.read_csv(path, sep='\t')

    train, rest = train_test_split(df, test_size=0.2, random_state=random_state)
    val, test = train_test_split(rest, test_size=0.5, random_state=random_state)

    # write train, val, test
    train.to_csv(dir_name + os.path.sep + file_name + '-train.tsv', sep='\t', index=False)
    val.to_csv(dir_name + os.path.sep + file_name + '-val.tsv', sep='\t', index=False)
    test.to_csv(dir_name + os.path.sep + file_name + '-test.tsv', sep='\t', index=False)


def create_dataset(df, author_ids, nlp):
    dataset = qa_from_author(df, nlp, author_ids[0])
    for author_id in author_ids[1:]:
        dataset = pd.concat([dataset, qa_from_author(df, nlp, author_id)])
    return dataset


def create_and_write_dataset(df, nlp, author_id, path):
    """
    Creates tsv dataset which contains only Apple support conversations with customers.
    """
    dataset = create_dataset(df, [author_id], nlp)
    dataset_path = path + author_id.lower() + '.tsv'
    dataset.to_csv(dataset_path, sep='\t', index=False)
    split_dataset(dataset_path)


def create_all_dataset(df, nlp, path):
    """
    Creates tsv dataset which contains many customer support services from dataset. Included support service authors are
    'AppleSupport', 'AmazonHelp', 'Uber_Support', 'Delta', 'SpotifyCares', 'Tesco', 'AmericanAir',
                  'comcastcares', 'TMobileHelp', 'British_Airways', 'SouthwestAir', 'Ask_Spectrum' and  'hulu_support'
    """
    author_ids = ['AppleSupport', 'AmazonHelp', 'Uber_Support', 'Delta', 'SpotifyCares', 'Tesco', 'AmericanAir',
                  'comcastcares', 'TMobileHelp', 'British_Airways', 'SouthwestAir', 'Ask_Spectrum', 'hulu_support']
    dataset = create_dataset(df, author_ids, nlp)
    dataset = dataset.sample(frac=1)  # shuffle dataset
    dataset_path = path + 'twitter-all' + '.tsv'
    dataset.to_csv(dataset_path, sep='\t', index=False)
    split_dataset(dataset_path)


def main():
    df = pd.read_csv(DATA_FOLDER + 'twcs.csv')
    df.sort_values(by='tweet_id', inplace=True)

    nlp = spacy.load('en')
    nlp.add_pipe(LanguageDetector())

    create_and_write_dataset(df, nlp, 'AppleSupport', DATA_FOLDER)
    create_and_write_dataset(df, nlp, 'AmazonHelp', DATA_FOLDER)
    create_and_write_dataset(df, nlp, 'Uber_Support', DATA_FOLDER)
    create_and_write_dataset(df, nlp, 'Delta', DATA_FOLDER)
    create_and_write_dataset(df, nlp, 'SpotifyCares', DATA_FOLDER)


if __name__ == '__main__':
    main()
