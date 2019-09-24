import os
import torch
from os.path import isfile, join

def ensure_exists(folder):
    """If the folder does not exist, create it.
    
    Arguments:
        folder {string} -- Folder.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

def ensure_npz(filename, inverse=False):
    if inverse:
        if filename.endswith('.npz'):
            filename = filename[:-4]
    else:
        if not filename.endswith('.npz'):
            filename = filename + '.npz'
    return filename

def ensure_tar(filename, inverse=False):
    if inverse:
        if filename.endswith('.tar'):
            filename = filename[:-4]
    else:
        if not filename.endswith('.tar'):
            filename = filename + '.tar'
    return filename

def list_files(folder):
    return [f for f in os.listdir(folder) if isfile(join(folder, f))]

def test_finiteness(tensor, description):
    if (~torch.isfinite(tensor)).sum() > 0:
        print('{}: NOT FINITE!'.format(description))