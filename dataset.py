from datasets.twitter_customer_support import load_dataset as twitter_dataset

dataset_map = {
    'twitter-customer-support': twitter_dataset
}


def dataset_factory(dataset_key, args, device):
    if dataset_key not in dataset_map:
        raise ValueError("There is no \"%s\" dataset, available datasets are: (%s)"
                         % (dataset_key, ', '.join(dataset_map.keys())))
    return dataset_map[dataset_key](args, device)

