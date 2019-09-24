import sys
import os
import random

import numpy as np
import kaldi.util.io as kio

def count_total_number_of_active_frames(vad_rxspecifiers):
    """Counts the total number of active speech frames in the given utterance list.
    
    Arguments:
        vad_rxspecifiers {list} -- List of lines of vad.scp file excluding utterance IDs.
    
    Returns:
        int -- Total number of active speech frames
        ndarray -- 1D array of frame indices that separate different utterances from each other (includes 0 as the first element and the total number of frames as the last element).
    """
    n_frames = 0
    counts = []
    for vad_specifier in vad_rxspecifiers:
        vad_labels = kio.read_vector(vad_specifier)
        n_active = np.sum(vad_labels.numpy().astype(int))
        counts.append(n_active)
        n_frames += n_active
    break_points = np.concatenate((np.atleast_1d(np.asarray(0, dtype=int)), np.cumsum(np.asarray(counts), dtype=int)))
    return n_frames, break_points

def load_posterior_specifiers(scp_file_without_ext):
    """Loads posterior reading specifiers from scp file.
    
    Arguments:
        scp_file_without_ext {string} -- Filename of scp file without the extension.
    
    Returns:
        list -- List of posterior reading specifiers.
    """
    scp_file = scp_file_without_ext + '.scp'
    rxspecifiers = []
    with open(scp_file) as f:
        for line in f:
            rxspecifiers.append(line.split()[-1].strip())
    return rxspecifiers


def _get_kaldi_dataset_files(folder):
    """Forms full filenames for feats.scp, vad.scp, utt2num_frames, and utt2spk files.
    
    Arguments:
        folder {string} -- Folder where the files are.
    
    Returns:
        (string, string, string, string) -- Full filenames for feats.scp, vad.scp, utt2num_frames, and utt2spk files, respectively.
    """
    feat_scp_file = os.path.join(folder, 'feats.scp')
    vad_scp_file = os.path.join(folder, 'vad.scp')
    utt2num_frames_file = os.path.join(folder, 'utt2num_frames')
    utt2spk_file = os.path.join(folder, 'utt2spk')
    return feat_scp_file, vad_scp_file, utt2num_frames_file, utt2spk_file


def _choose_utterances(data_folder, meta_folder, selected_utts):
    """Loads selected utterance and speaker IDs and feature and vad reading specifiers from the given folder (meta_folder). Fixes specifiers to point to ark files that have been moved from their original location.
    
    Arguments:
        data_folder {string} -- Used to fix specifiers to point to correct ark files in case ark files were moved from their original location. Last subfolder of data_folder should have the same name as in the original path to the ark files.
        meta_folder {string} -- Folder where the feats.scp, vad.scp, utt2num_frames, and utt2spk are.
        selected_utts {set} -- Set of utterance IDs that should be selected. If None, selects all. 
    
    Returns:
        list -- Reading specifiers for features.
        list -- Reading specifiers for VAD labels.
        list -- Utterance IDs.
        list -- Speaker IDs.
    """
    feat_scp_file, vad_scp_file, utt2num_frames_file, utt2spk_file = _get_kaldi_dataset_files(meta_folder)
    base_folder = os.sep + os.path.basename(os.path.normpath(data_folder)) + os.sep
    feat_rxfilenames = []
    vad_rxfilenames = []
    utts = []
    spks = []
    with open(feat_scp_file) as f1, open(vad_scp_file) as f2, open(utt2spk_file) as f3:
        for line1, line2, line3 in zip(f1, f2, f3):
            parts1 = line1.split()
            if selected_utts is None or parts1[0] in selected_utts:
                parts2 = line2.split()
                parts3 = line3.split()
                if parts1[0] != parts2[0] or parts1[0] != parts3[0]:
                    sys.exit('Error: scp-files are not aligned!')
                feat_loc = parts1[1].split(base_folder)[1].strip()
                vad_loc = parts2[1].split(base_folder)[1].strip()
                feat_rxfilenames.append(os.path.join(data_folder, feat_loc))
                vad_rxfilenames.append(os.path.join(data_folder, vad_loc))
                utts.append(parts1[0])
                spks.append(parts3[1].strip())
    return feat_rxfilenames, vad_rxfilenames, utts, spks


def choose_all(data_folder, meta_folder):
    """Loads all utterance and speaker IDs and feature and vad reading specifiers from the given folder (meta_folder). Fixes specifiers to point to ark files that have been moved from their original location.
    
    Arguments:
        data_folder {string} -- Used to fix specifiers to point to correct ark files in case ark files were moved from their original location. Last subfolder of data_folder should have the same name as in the original path to the ark files.
        meta_folder {string} -- Folder where the feats.scp, vad.scp, utt2num_frames, and utt2spk are.
    
    Returns:
        list -- Reading specifiers for features.
        list -- Reading specifiers for VAD labels.
        list -- Utterance IDs.
        list -- Speaker IDs.
    """
    print('Loading all feature-specifiers, utterance labels, and speaker labels from folder {}'.format(meta_folder))
    return _choose_utterances(data_folder, meta_folder, None)


def choose_n_longest(data_folder, meta_folder, n):
    """Same as choose_all function with a difference that this functions chooses the n longest utterances from the specified folder.
    """
    print('Loading feature-specifiers, utterance labels, and speaker labels of the {} longest files from folder {}'.format(n, meta_folder))
    feat_scp_file, vad_scp_file, utt2num_frames_file, utt2spk_file = _get_kaldi_dataset_files(meta_folder)
    selected_utts = set()
    utts = []
    num_frames = []
    with open(utt2num_frames_file) as f:
        for line in f:
            parts = line.split()
            utts.append(parts[0])
            num_frames.append(int(parts[1].strip()))
    num_frames = np.asarray(num_frames, dtype=int)
    indices = np.argsort(num_frames)
    indices = indices[-n:]       
    for index in indices:
        selected_utts.add(utts[index])
    return _choose_utterances(data_folder, meta_folder, selected_utts)


def choose_n_random(data_folder, meta_folder, n, seed=0):
    """Same as choose_all function with a difference that this functions chooses n random utterances from the specified folder.
    """
    random.seed(seed)
    print('Loading feature-specifiers, utterance labels, and speaker labels of {} random files from folder {}'.format(n, meta_folder))
    feat_scp_file, vad_scp_file, utt2num_frames_file, utt2spk_file = _get_kaldi_dataset_files(meta_folder)
    utts = []
    with open(utt2num_frames_file) as f:
        for line in f:
            parts = line.split()
            utts.append(parts[0])
    return _choose_utterances(data_folder, meta_folder, set(random.sample(utts, n)))


def choose_from_wavfile(data_folder, meta_folder, wav_file, every_nth=1):
    """Chooses every n:th utterance from the wav_file.
    """
    print('Loading (every {}) feature-specifiers, utterance labels, and speaker labels for utterances in {} from folder {}'.format(every_nth, wav_file, meta_folder))

    utts = set()
    with open(wav_file) as f:
        for index, line in enumerate(f):
            if index % every_nth == 0:
                parts = line.split()
                utts.add(parts[0])
    return _choose_utterances(data_folder, meta_folder, utts)