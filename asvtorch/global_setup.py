# Do not touch this file directly. Change the device from run_voxceleb_ivector.py.

import torch

device = torch.device("cpu")

def set_gpu(device_id):
    if torch.cuda.is_available():
        global device
        device = torch.device('cuda:{}'.format(device_id))
        torch.backends.cudnn.benchmark = False
        print('Using GPU!')
    else:
        print('Cuda is not available!')
