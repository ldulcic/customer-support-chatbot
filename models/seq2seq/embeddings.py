import torch.nn as nn


def embeddings_factory(args, metadata):
    embed = nn.Embedding(num_embeddings=metadata.vocab_size, embedding_dim=args.embedding_size,
                         padding_idx=metadata.padding_idx, _weight=metadata.vectors)
    # whether we will propagate gradients to word embeddings
    embed.weight.require_grads = args.train_embeddings
    return embed
