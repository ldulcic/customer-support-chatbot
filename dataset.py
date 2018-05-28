from datasets.twitter_customer_support.dataset import load_dataset as twitter_dataset

dataset_map = {
    'twitter-apple': twitter_dataset,
    'twitter-small': twitter_dataset
}


def dataset_factory(args, device):
    if args.dataset not in dataset_map:
        raise ValueError("There is no \"%s\" dataset, available datasets are: (%s)"
                         % (args.dataset, ', '.join(dataset_map.keys())))
    return dataset_map[args.dataset](args, device)

