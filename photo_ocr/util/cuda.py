import torch

import torch.backends.cudnn as cudnn

CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if CUDA else 'cpu')

CUDA = False
DEVICE = "cpu"

if CUDA:
    cudnn.benchmark = True
    cudnn.deterministic = True
