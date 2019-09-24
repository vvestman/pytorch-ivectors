import numpy as np
from kaldi.util.table import VectorWriter
import kaldi.util.io as kio
from kaldi.matrix import Vector

class PosteriorWriter():
    def __init__(self, wxspecifier):
        self.posterior_writer = VectorWriter(wxspecifier)

    def write(self, utt_id, counts, posteriors, indices):
        """Writes posteriors to disk in KALDI format.
        
        Arguments:
            utt_id {string} -- Utterance ID to be written to scp file
            counts {Tensor} -- Tensor containing the numbers of selected posteriors for each frame
            posteriors {Tensor} -- Flattened Tensor containing all posteriors
            indices {Tensor} -- Flattened Tensor containing all Gaussian indices
        """

        counts = counts.numpy()
        posteriors = posteriors.numpy()
        indices = indices.numpy()
        nframes = np.atleast_1d(np.array([counts.size]))
        datavector = np.hstack([nframes, counts, posteriors, indices])
        datavector = Vector(datavector)
        self.posterior_writer.write(utt_id, datavector)       

    def close(self):
        self.posterior_writer.close()


def load_posteriors(rxspecifier):
    """Loads posteriors stored in KALDI format from disk.

    Arguments:
        rxspecifier {string} -- A line from scp file excluding the utterance ID.
    
    Returns:
        ndarray -- Array containing the numbers of selected posteriors for each frame
        ndarray -- Array containing posteriors (flattened)
        ndarray -- Array containing Gaussian indices (flattened)
    """

    datavector = kio.read_vector(rxspecifier)
    datavector = datavector.numpy()
    nframes = int(datavector[0])
    counts = datavector[1:nframes+1].astype(int)
    n_posteriors = (datavector.size - counts.size - 1) // 2
    posteriors = datavector[nframes+1:-n_posteriors]
    indices = datavector[-n_posteriors:].astype(int)
    return counts, posteriors, indices