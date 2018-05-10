import pandas as pd
import preprocessor as p
import emoji
import re
import spacy
from spacy_cld import LanguageDetector


def id2text(df, x):
    row = df[df.tweet_id == int(x)]
    # some ids are missing, they just don't exist in data
    return '' if (len(row) == 0) else row.iloc[0]['text']


def clean_tweet(tweet):
    # TODO think about not removing numbers when preprocessing (uncomment line below)
    # p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.MENTION, p.OPT.HASHTAG)
    # TODO think about replacing hashtags/emojis/etc. with specials word using p.tokenize
    # removes @ mentions, urls, hashtags and emojis
    clean = p.clean(tweet)

    # preprocessor doesn't seem to clean all emojis so we run text trough emoji regex to clean lefovers
    clean = re.sub(emoji.get_emoji_regexp(), '', clean)

    # removing zero-width character which is often bundled with emojis
    clean = re.sub(u'\ufe0f', '', clean)

    # remove multiple empty spaces with one
    clean = re.sub(r' +', ' ', clean)

    # strip any leftover spaces at the beginning and end
    clean = clean.strip()

    return clean


def set_empty_if_not_english(nlp, x):
    doc = nlp(x)
    # TODO tweets with no language tend to be valid, but they also tend to be gibberish, filtering for now
    # TODO don't filter scotish, don't filter danish with "icloud" in it (for some reason every tweet with icloud in it is classified as danish), filtering for now
    return x if doc._.languages and ('en' in doc._.language_scores and doc._.language_scores['en'] >= 0.80) else ''


def qa_from_author(df, author_id):
    """
    Creates qa dataset (in from of dataframe) from all tweets of author (identified by author_id)

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
    support_service = support_service[['text', 'in_response_to_tweet_id']]

    # replace tweet ids with actual tweet text
    support_service.loc[:, 'in_response_to_tweet_id'] = support_service.in_response_to_tweet_id.apply(lambda x: id2text(df, x))

    # rename and rearrange columns
    support_service.rename(columns={'text': 'answer', 'in_response_to_tweet_id': 'question'}, inplace=True)
    support_service = support_service[['question', 'answer']]

    # clean twitter data
    support_service.loc[:, 'question'] = support_service.question.apply(clean_tweet)
    support_service.loc[:, 'answer'] = support_service.answer.apply(clean_tweet)

    # filter all languages which are not english (non-english tweets will be set to empty string and then filtered at
    # the end of this method)
    nlp = spacy.load('en')
    nlp.add_pipe(LanguageDetector())
    support_service.loc[:, 'question'] = support_service.question.apply(lambda x: set_empty_if_not_english(nlp, x))
    support_service.loc[:, 'answer'] = support_service.answer.apply(lambda x: set_empty_if_not_english(nlp, x))

    # remove all QA pairs where Q or A are empty or contain only dot (.)
    support_service = support_service[~(support_service.question == '') & ~(support_service.answer == '')]
    support_service = support_service[~(support_service.question == '.') & ~(support_service.answer == '.')]

    return support_service


def create_dataset(df, author_id, path):
    apple_support = qa_from_author(df, author_id)
    apple_support.to_csv(path + author_id.lower() + '.tsv', sep='\t', index=False)


def main():
    df = pd.read_csv('../../data/twitter_customer_support/twcs.csv')
    df.sort_values(by='tweet_id', inplace=True)

    create_dataset(df, 'AppleSupport', '../../data/twitter_customer_support/')


if __name__ == '__main__':
    main()
