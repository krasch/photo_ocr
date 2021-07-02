import torch

import torch.backends.cudnn as cudnn

CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if CUDA else 'cpu')

if CUDA:
    cudnn.benchmark = True
    cudnn.deterministic = True


RECOGNITION_BATCH_SIZE = 2
