from datasets.twitter_customer_support import load_dataset as twitter_dataset

dataset_map = {
    'twitter-customer-support': twitter_dataset
}


def dataset_factory(dataset):
    if dataset not in dataset_map:
        raise ValueError("There is no \"%s\" dataset, available datasets are: (%s)"
                         % (dataset, ', '.join(dataset_map.keys())))
    return dataset_map[dataset]()


if __name__ == '__main__':
    dataset_factory('twitter-customer-support')
