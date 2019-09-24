import time

import numpy as np
import torch
from kaldi.util.table import VectorWriter
from kaldi.matrix import Vector

import asvtorch.ivector.featureloader
import asvtorch.global_setup as gs
from asvtorch.ivector.gmm import DiagGmm
from asvtorch.kaldidata.posterior_io import PosteriorWriter
from asvtorch.misc.misc import ensure_npz


def batch_extract_posteriors(rx_specifiers, utt_ids, feature_loader, ubm, output_filename, settings):
    """Extracts posteriors using full covariance matrices. Computational requirements and disk space requirements are reduced by performing Gaussian selection using diagonal covariances and by thresholding posteriors.
    
    Arguments:
        rx_specifiers {(list, list)} --  Two lists in a tuple containing scp lines without utterance IDs for features and VAD labels, respectively.
        utt_ids {list} -- Utterance IDs.
        feature_loader {KaldiFeatureLoader} -- Feature loader.
        ubm {Gmm} -- A GMM (UBM).
        output_filename {string} -- Output filename for posteriors (without extension).
        settings {PosteriorExtractionSettings} - Settings.
    """
    
    print('Extracting posteriors for {} utterances...'.format(len(rx_specifiers[0])))

    dataloader = asvtorch.ivector.featureloader.get_feature_loader(rx_specifiers, feature_loader, settings.batch_size_in_frames, settings.dataloader_workers)

    diag_ubm = DiagGmm.from_full_gmm(ubm, gs.device)

    sub_batch_count = int(np.ceil(ubm.means.size()[0] / ubm.means.size()[1]))

    wspecifier_top_posterior = "ark,scp:{0}.ark,{0}.scp".format(output_filename)
    posterior_writer = PosteriorWriter(wspecifier_top_posterior)

    posterior_buffer = torch.Tensor()
    top_buffer = torch.LongTensor()
    count_buffer = torch.LongTensor()

    start_time = time.time()
    frame_counter = 0
    utterance_counter = 0

    start_time = time.time()
    for batch_index, batch in enumerate(dataloader):

        frames, end_points = batch
        frames = frames.to(gs.device)
        frames_in_batch = frames.size()[0]

        chunks = torch.chunk(frames, sub_batch_count, dim=0)
        top_gaussians = []
        for chunk in chunks:
            posteriors = diag_ubm.compute_posteriors(chunk)  
            top_gaussians.append(torch.topk(posteriors, settings.n_top_gaussians, dim=0, largest=True, sorted=False)[1])
        
        top_gaussians = torch.cat(top_gaussians, dim=1)
        
        posteriors = ubm.compute_posteriors_top_select(frames, top_gaussians)

        # Posterior thresholding:
        max_indices = torch.argmax(posteriors, dim=0)
        mask = posteriors.ge(settings.posterior_threshold)
        top_counts = torch.sum(mask, dim=0)
        posteriors[~mask] = 0
        divider = torch.sum(posteriors, dim=0)
        mask2 = divider.eq(0) # For detecting special cases
        posteriors[:, ~mask2] = posteriors[:, ~mask2] / divider[~mask2]
        # Special case that all the posteriors are discarded (force to use 1):
        posteriors[max_indices[mask2], mask2] = 1 
        mask[max_indices[mask2], mask2] = 1 
        top_counts[mask2] = 1

        # Vectorize the data & move to cpu memory
        posteriors = posteriors.t().masked_select(mask.t())
        top_gaussians = top_gaussians.t().masked_select(mask.t())
        posteriors = posteriors.cpu()
        top_gaussians = top_gaussians.cpu()
        top_counts = top_counts.cpu()

        end_points = end_points - frame_counter  # relative end_points in a batch

        if end_points.size != 0:
            # Save utterance data that continues from the previous batch:
            psave = torch.cat([posterior_buffer, posteriors[:torch.sum(top_counts[:end_points[0]])]])
            tsave = torch.cat([top_buffer, top_gaussians[:torch.sum(top_counts[:end_points[0]])]])
            csave = torch.cat([count_buffer, top_counts[:end_points[0]]])
            posterior_writer.write(utt_ids[utterance_counter], csave, psave, tsave)
            utterance_counter += 1

            # Save utterance data that is fully included in this batch:
            for start_point, end_point in zip(end_points[:-1], end_points[1:]):
                psave = posteriors[torch.sum(top_counts[:start_point]):torch.sum(top_counts[:end_point])]
                tsave = top_gaussians[torch.sum(top_counts[:start_point]):torch.sum(top_counts[:end_point])]
                csave = top_counts[start_point:end_point]
                posterior_writer.write(utt_ids[utterance_counter], csave, psave, tsave)
                utterance_counter += 1
            
            # Buffer partial data to be used in the next batch:
            posterior_buffer = posteriors[torch.sum(top_counts[:end_points[-1]]):]
            top_buffer = top_gaussians[torch.sum(top_counts[:end_points[-1]]):]
            count_buffer = top_counts[end_points[-1]:]
        else:
            # Buffer the whole data for the next batch (if the utterance is longer than the current batch (special case)):
            posterior_buffer = torch.cat([posterior_buffer, posteriors])
            top_buffer = torch.cat([top_buffer, top_gaussians])
            count_buffer = torch.cat([count_buffer, top_counts])

        frame_counter += frames_in_batch

        print('{:.0f} seconds elapsed, batch {}/{}: {}, utterance count (roughly) = {}'.format(time.time() - start_time, batch_index+1, dataloader.__len__(), frames.size(), len(end_points)))

    posterior_writer.close()
    print('Posterior computation completed in {:.3f} seconds'.format(time.time() - start_time))

    return time.time() - start_time
