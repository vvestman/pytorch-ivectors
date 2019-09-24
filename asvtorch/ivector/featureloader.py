import numpy as np
import torch
from torch.utils import data

from asvtorch.kaldidata.utils import count_total_number_of_active_frames


def _get_clip_indices(utt_start, utt_end, batch_start, batch_end):    
    """ Cuts the parts of the utterance that do not fit into the batch window.
    
    Arguments:
        utt_start {int} -- start point of the utterance
        utt_end {int} -- end point of the utterance
        batch_start {int} -- start point of the batch window
        batch_end {int} -- end point of the batch window
    
    Returns:
        (int, int), bool -- a tuple containing clipped start and end point of an utterance, the boolean flag is True if the end of the utterance is inside the batch window.
    """
    if utt_end <= batch_start:
        return None
    if utt_start >= batch_end:
        return None
    start = 0
    end = utt_end - utt_start
    if utt_start < batch_start:
        start = batch_start - utt_start
    if utt_end > batch_end:
        end = batch_end - utt_start
    if utt_end <= batch_end:
        ends = True
    else:
        ends = False
    return (start, end), ends

class _Kaldi_Dataset(data.Dataset):
    def __init__(self, rxspecifiers, feature_loader, frames_per_batch):
        self.feat_rxspecifiers = rxspecifiers[0]
        self.vad_rxspecifiers = rxspecifiers[1]
        self.feature_loader = feature_loader
      
        n_active_frames, break_points = count_total_number_of_active_frames(self.vad_rxspecifiers)
        n_batches = int(np.ceil(n_active_frames / frames_per_batch))

        utt_index = 0
        self.batches = []

        for i in range(n_batches):
            batch_indices = []
            batch_endpoints = []
            window_start = i * frames_per_batch
            window_end = (i + 1) * frames_per_batch
            while utt_index < len(self.feat_rxspecifiers):
                clip_indices = _get_clip_indices(break_points[utt_index], break_points[utt_index + 1], window_start, window_end)
                utt_index += 1
                if clip_indices is not None:
                    batch_indices.append((utt_index - 1, clip_indices[0]))
                    if clip_indices[1]:
                        batch_endpoints.append(break_points[utt_index])
                    else:
                        break
                else:
                    if batch_indices:
                        break
            self.batches.append((batch_indices, np.asarray(batch_endpoints)))
            batch_indices = []
            batch_endpoints = []
            utt_index -= 1

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        batch_indices, batch_endpoints = self.batches[index]
        frames = []
        for utt_indices in batch_indices:
            utt_index, selection_indices = utt_indices
            feats = self.feature_loader.load_features(self.feat_rxspecifiers[utt_index], self.vad_rxspecifiers[utt_index])
            frames.append(feats[selection_indices[0]:selection_indices[1], :])        
        frames = torch.from_numpy(np.vstack(frames))
        return frames, batch_endpoints


def _collater(batch):
    """ In this "hack" batches are already formed in the DataSet object (batch consists of a single element, which is actually the batch). 
    """
    return batch[0]

def get_feature_loader(rxspecifiers, feature_loader, batch_size, num_workers):
    """Returs a DataLoader that is used to load features from multiple utterances using a fixed batch size given in (active speech) frames.
    
    Arguments:
        rxspecifiers {(list, list)} -- [description]
        feature_loader {KaldiFeatureLoader} -- Feature loader.
        batch_size {int} -- Batch size in frames.
        num_workers {int} -- Number of processes used for data loading.
    
    Returns:
        DataLoader -- DataLoader for reading features.
    """
    dataset = _Kaldi_Dataset(rxspecifiers, feature_loader, batch_size)
    return data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=_collater)
