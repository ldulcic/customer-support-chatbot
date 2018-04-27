from models.seq2seq import Encoder, Decoder, Seq2Seq


def model_factory(args, field, vocab_size, padding_idx):
    # init encoder and decoder
    encoder = Encoder(vocab_size=vocab_size, embed_size=args.embedding_size, hidden_size=args.encoder_hidden_size,
                      padding_idx=padding_idx, num_layers=args.encoder_num_layers, bidirectional=args.encoder_bidirectional)
    decoder = Decoder(vocab_size=vocab_size, embed_size=args.embedding_size, hidden_size=args.decoder_hidden_size,
                      padding_idx=padding_idx, num_layers=args.decoder_num_layers)

    # optionally load pre-trained embeddings
    if args.embedding_type:
        encoder.embed.weight.data.copy_(field.vocab.vectors)
        decoder.embed.weight.data.copy_(field.vocab.vectors)

    # whether we will propagate gradients to word embeddings
    encoder.embed.weight.require_grads = args.train_embeddings
    decoder.embed.weight.require_grads = args.train_embeddings

    return Seq2Seq(encoder, decoder, vocab_size)
