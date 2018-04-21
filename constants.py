import torch


SOS_TOKEN = '<sos>'  # start of sentence token
EOS_TOKEN = '<eos>'  # end of sentence token
CUDA = torch.cuda.is_available()
