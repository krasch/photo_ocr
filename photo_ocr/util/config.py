import torch

import torch.backends.cudnn as cudnn


class Config:
    # do private member variables to deter library users from directly manipulating the values
    # instead, this class should only be used through it's getters and setters
    # (which is a bit of a pain because need to call config.get_device() all over the place,
    #  but worth it because less error prone for end user )
    _device = None
    _recognition_batch_size = 32

    def __init__(self):
        # GPU is available, init to use default GPU
        if torch.cuda.is_available():
            self.set_device("cuda")  # "cuda" = default GPU
            cudnn.benchmark = True
            cudnn.deterministic = True

        # no GPU is available, fall back to CPU
        else:
            self.set_device("cpu")

    def set_device(self, device: str):
        """
        Set the device to be used by pytorch
        :param device: e.g. "cpu", "cuda", "cuda:0", "cuda:1", etc
        :return:
        """
        self._device = torch.device(device)

    def set_recognition_batch_size(self, batch_size: int):
        self._recognition_batch_size = batch_size

    def get_device(self):
        return self._device

    def get_recognition_batch_size(self):
        return self._recognition_batch_size


# init config singleton object; this will also init the GPU (if available)
config = Config()

