import torch
import numpy as np

def organize_trials(vectors, utt_ids, trial_file):
    trial_vector_dict = {}
    for index, segment in enumerate(utt_ids):
        trial_vector_dict[segment] = vectors[index, :]

    trials = []
    with open(trial_file) as f:
        for line in f:
            parts = line.split()
            if parts[2].strip() == 'target':
                label = 1
            else:
                label = 0
            trials.append((parts[0], parts[1], label))

    left_vectors = torch.zeros(len(trials), vectors.shape[1], device=vectors.device)
    right_vectors = torch.zeros(len(trials), vectors.shape[1], device=vectors.device)

    labels = []
    for index, trial in enumerate(trials):
        left_vectors[index, :] = trial_vector_dict[trial[0]]
        right_vectors[index, :] = trial_vector_dict[trial[1]]
        labels.append(trial[2])

    labels = np.asarray(labels, dtype=bool)

    return left_vectors, right_vectors, labels


def organize_trials_in_chunks(vectors, utt_ids, trial_file, chunk_size):
    
    print('Preparing to iterate over trials...')

    trial_vector_dict = {}
    for index, segment in enumerate(utt_ids):
        trial_vector_dict[segment] = vectors[index, :]

    trials = []
    with open(trial_file) as f:
        for line in f:
            parts = line.split()
            if parts[2].strip() == 'target':
                label = 1
            else:
                label = 0
            trials.append((parts[0], parts[1], label))

    i = 0

    while i < len(trials):
        print('Iterated over {} trials'.format(i))
        chunk_trials = trials[i:i+chunk_size]
        i += chunk_size

        left_vectors = torch.zeros(len(chunk_trials), vectors.shape[1], device=vectors.device)
        right_vectors = torch.zeros(len(chunk_trials), vectors.shape[1], device=vectors.device)

        labels = []
        for index, trial in enumerate(chunk_trials):
            left_vectors[index, :] = trial_vector_dict[trial[0]]
            right_vectors[index, :] = trial_vector_dict[trial[1]]
            labels.append(trial[2])

        labels = np.asarray(labels, dtype=bool)

        yield left_vectors, right_vectors, labels
