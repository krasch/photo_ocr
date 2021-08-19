import torch

import torch.backends.cudnn as cudnn


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    cudnn.benchmark = True
    cudnn.deterministic = True

else:
    DEVICE = torch.device("cpu")

