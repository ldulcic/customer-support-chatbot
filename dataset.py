from datasets.twitter_customer_support.dataset import load_dataset as twitter_dataset
from datasets.twitter_customer_support.dataset import load_field as twitter_field
from datasets.twitter_customer_support.dataset import load_metadata as twitter_metadata


DATASET_IDX = 0
FIELD_IDX = 1
METADATA_IDX = 2

dataset_field_map = {
    'twitter-applesupport': (twitter_dataset, twitter_field, twitter_metadata),
    'twitter-amazonhelp': (twitter_dataset, twitter_field, twitter_metadata),
    'twitter-delta': (twitter_dataset, twitter_field, twitter_metadata),
    'twitter-spotifycares': (twitter_dataset, twitter_field, twitter_metadata),
    'twitter-uber_support': (twitter_dataset, twitter_field, twitter_metadata),
    'twitter-all': (twitter_dataset, twitter_field, twitter_metadata),
    'twitter-small': (twitter_dataset, twitter_field, twitter_metadata)
}


def get_dataset_tuple(args):
    if args.dataset not in dataset_field_map:
        raise ValueError("There is no \"%s\" dataset, available datasets are: (%s)"
                         % (args.dataset, ', '.join(dataset_field_map.keys())))
    return dataset_field_map[args.dataset]


def dataset_factory(args, device):
    dataset_tuple = get_dataset_tuple(args)
    return dataset_tuple[DATASET_IDX](args, device)


def field_factory(args):
    dataset_tuple = get_dataset_tuple(args)
    return dataset_tuple[FIELD_IDX]()


def metadata_factory(args, vocab):
    dataset_tuple = get_dataset_tuple(args)
    return dataset_tuple[METADATA_IDX](vocab)
