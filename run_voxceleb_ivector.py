import os
import socket
import datetime

import torch
import numpy as np

import asvtorch.kaldidata.kaldifeatloaders
import asvtorch.kaldidata.utils
from asvtorch.misc.misc import ensure_exists
import asvtorch.ivector.posteriors
import asvtorch.ivector
import asvtorch.ivector.ivector_extractor
import asvtorch.ivector.settings
import asvtorch.evaluation.trials
import asvtorch.evaluation.eval_metrics
from asvtorch.ivector.gmm import Gmm
from asvtorch.backend.plda import Plda
from asvtorch.backend.vector_processing import VectorProcessor
from asvtorch.evaluation.parameters import ParameterChanger

# UPDATE THIS TO THE SAME FOLDER THAT WAS USED IN THE KALDI SCRIPT FOR OUTPUTS:
DATA_FOLDER = '/media/hdd2/vvestman/voxceleb_outputs'

TRY_TO_USE_GPU = True

if TRY_TO_USE_GPU:
    if torch.cuda.is_available():
        asvtorch.global_setup.device = torch.device("cuda:0")
        print('Using GPU!')

print('Loading settings...')
posterior_extraction_settings = asvtorch.ivector.settings.PosteriorExtractionSettings()
posterior_extraction_settings.dataloader_workers = 4
ivector_settings = asvtorch.ivector.settings.IVectorSettings()

class RecipeSettings():
    def __init__(self):
        self.start_stage = 0
        self.end_stage = 3
        self.plda_dim = 200
recipe_settings = RecipeSettings()


posterior_extraction_settings.print_settings()
ivector_settings.print_settings()

parameter_changer = ParameterChanger('config.py', {'ivector': ivector_settings, 'recipe': recipe_settings})
ensure_exists(os.path.join(DATA_FOLDER, 'results'))
resultfile = open(os.path.join(DATA_FOLDER, 'results', 'results_{}.txt'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))), 'w')


# Input data:
TRAIN_FOLDER = os.path.join(DATA_FOLDER, 'train')
TEST_FOLDER = os.path.join(DATA_FOLDER, 'voxceleb1_test')
TRIAL_FILE = os.path.join(TEST_FOLDER, 'trials')
UBM_FILE = os.path.join(DATA_FOLDER, 'full_ubm', 'final.ubm')
VOX1_TRAIN_WAVFILE = os.path.join(DATA_FOLDER, 'voxceleb1_train', 'wav.scp')

# Output files:
IVEC_TRAINING_POSTERIOR_FILE = os.path.join(DATA_FOLDER, 'posteriors', 'ivec_posteriors')
BACKEND_TRAINING_POSTERIOR_FILE = os.path.join(DATA_FOLDER, 'posteriors', 'backend_posteriors')
TESTING_POSTERIOR_FILE = os.path.join(DATA_FOLDER, 'posteriors', 'testing_posteriors')
ensure_exists(os.path.join(DATA_FOLDER, 'posteriors'))

EXTRACTOR_OUTPUT_FILE = os.path.join(DATA_FOLDER, 'iextractor', 'iextractor')
ensure_exists(os.path.join(DATA_FOLDER, 'iextractor'))


print('Initializing feature loader...')
feature_loader = asvtorch.kaldidata.kaldifeatloaders.VoxcelebFeatureLoader()
print('Loading KALDI UBM...')
ubm = Gmm.from_kaldi(UBM_FILE, asvtorch.global_setup.device)


# Dataset preparation
print('Choosing dataset for i-vector extractor training...')
feat_rxspecifiers, vad_rxspecifiers, utt_ids, spk_ids = asvtorch.kaldidata.utils.choose_n_longest(DATA_FOLDER, TRAIN_FOLDER, 100000)
rxspecifiers = (feat_rxspecifiers, vad_rxspecifiers)

print('Choosing dataset for PLDA training...')
plda_feat_rxspecifiers, plda_vad_rxspecifiers, plda_utt_ids, plda_spk_ids = asvtorch.kaldidata.utils.choose_from_wavfile(DATA_FOLDER, TRAIN_FOLDER, VOX1_TRAIN_WAVFILE, 1)
plda_rxspecifiers = (plda_feat_rxspecifiers, plda_vad_rxspecifiers)

test_feat_rxspecifiers, test_vad_rxspecifiers, test_utt_ids, test_spk_ids = asvtorch.kaldidata.utils.choose_all(DATA_FOLDER, TEST_FOLDER)
test_rxspecifiers = (test_feat_rxspecifiers, test_vad_rxspecifiers)


while parameter_changer.next(): 

    # Frame posterior extraction
    if recipe_settings.start_stage <= 1 <= recipe_settings.end_stage:
        asvtorch.ivector.posteriors.batch_extract_posteriors(rxspecifiers, utt_ids, feature_loader, ubm, IVEC_TRAINING_POSTERIOR_FILE, posterior_extraction_settings)
        asvtorch.ivector.posteriors.batch_extract_posteriors(plda_rxspecifiers, plda_utt_ids, feature_loader, ubm, BACKEND_TRAINING_POSTERIOR_FILE, posterior_extraction_settings)
        asvtorch.ivector.posteriors.batch_extract_posteriors(test_rxspecifiers, test_utt_ids, feature_loader, ubm, TESTING_POSTERIOR_FILE, posterior_extraction_settings)


    # Preparing data with posteriors
    posterior_rxspecifiers = asvtorch.kaldidata.utils.load_posterior_specifiers(IVEC_TRAINING_POSTERIOR_FILE)
    rxspecifiers = (*rxspecifiers, posterior_rxspecifiers)  # Tuple of three elements: (feats, vad, posteriors)
    plda_posterior_rxspecifiers = asvtorch.kaldidata.utils.load_posterior_specifiers(BACKEND_TRAINING_POSTERIOR_FILE)
    plda_rxspecifiers = (*plda_rxspecifiers, plda_posterior_rxspecifiers)  # Tuple of three elements: (feats, vad, posteriors)
    test_posterior_rxspecifiers = asvtorch.kaldidata.utils.load_posterior_specifiers(TESTING_POSTERIOR_FILE)
    test_rxspecifiers = (*test_rxspecifiers, test_posterior_rxspecifiers)  # Tuple of three elements: (feats, vad, posteriors)



    if recipe_settings.start_stage <= 2 <= recipe_settings.end_stage:
  
        # I-vector extractor training
        ivector_extractor = asvtorch.ivector.ivector_extractor.IVectorExtractor.random_init(ubm, ivector_settings, asvtorch.global_setup.device, seed=0)
        iteration_times = ivector_extractor.train(rxspecifiers, feature_loader, EXTRACTOR_OUTPUT_FILE, ivector_settings)

        for iteration in range(1, ivector_settings.n_iterations + 1):

            ivector_extractor = asvtorch.ivector.ivector_extractor.IVectorExtractor.from_npz(EXTRACTOR_OUTPUT_FILE, asvtorch.global_setup.device, iteration)
            
            # Extracting i-vectors
            plda_training_vectors = ivector_extractor.extract(plda_rxspecifiers, feature_loader, ivector_settings)
            test_vectors = ivector_extractor.extract(test_rxspecifiers, feature_loader, ivector_settings)
  
            # Processing i-vectors
            vector_processor = VectorProcessor.train(plda_training_vectors, 'cwl', asvtorch.global_setup.device)
            plda_training_vectors = vector_processor.process(plda_training_vectors)
            test_vectors = vector_processor.process(test_vectors)
            
            # Training PLDA
            plda = Plda.train_closed_form(plda_training_vectors, plda_spk_ids, asvtorch.global_setup.device)
        
            # Arranging trials
            left_vectors, right_vectors, labels = asvtorch.evaluation.trials.organize_trials(test_vectors, test_utt_ids, TRIAL_FILE)
            
            # Scoring
            scores = plda.score_trials(left_vectors, right_vectors, recipe_settings.plda_dim)
            eer = asvtorch.evaluation.eval_metrics.compute_eer(scores[labels], scores[~labels])[0]

            # Printing results
            print(parameter_changer.get_current_string(compact=False))
            print('EER: {:.4f} %'.format(eer*100))
            resultfile.write('{:.4f} {:.2f} {} {}\n'.format(eer*100, iteration_times[iteration-1], iteration, parameter_changer.get_value_string()))
            resultfile.flush()

resultfile.close()
