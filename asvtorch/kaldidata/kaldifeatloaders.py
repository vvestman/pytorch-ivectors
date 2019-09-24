import abc

import numpy as np

import kaldi.feat.functions as featfuncs
import kaldi.util.io as kio

class KaldiFeatureLoader(abc.ABC):
    @abc.abstractmethod
    def load_features(self, feat_rxspecifier, vad_rxspecifier):
        """Loads and processes KALDI features.
        
        Arguments:
            feat_rxspecifier {string} -- A line from feats.scp file excluding the utterance ID
            vad_rxspecifier {string} -- A line from vad.scp file excluding the utterance ID
        """
        pass


class VoxcelebFeatureLoader(KaldiFeatureLoader):
    """ This class is used to read features extracted by the KALDI recipe "egs/voxceleb/v1/". After loading the features are procesessed in the same manner as done in the KALDI recipe.
    """
    def __init__(self):
        self.delta_opts = featfuncs.DeltaFeaturesOptions(order=2, window=3)
        self.cmn_opts = featfuncs.SlidingWindowCmnOptions()
        self.cmn_opts.center = True
        self.cmn_opts.cmn_window = 300
        self.cmn_opts.normalize_variance = False

    def load_features(self, feat_rxspecifier, vad_rxspecifier):
        feats = kio.read_matrix(feat_rxspecifier)
        vad_labels = kio.read_vector(vad_rxspecifier)
        feats = featfuncs.compute_deltas(self.delta_opts, feats)
        featfuncs.sliding_window_cmn(self.cmn_opts, feats, feats)
        feats = feats.numpy()[vad_labels.numpy().astype(bool), :]
        return feats
