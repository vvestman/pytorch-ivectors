import numpy as np
import torch
import kaldi.util.io as kio
from kaldi.gmm import FullGmm as KaldiFullGmm
from kaldi.matrix import Matrix as KaldiMatrix

import asvtorch.global_setup as gs
from asvtorch.misc.misc import ensure_npz
from asvtorch.misc.misc import test_finiteness

class Gmm():
    def __init__(self, means, covariances, weights, device=torch.device("cpu")):
        self.means = means.to(device)
        self.covariances = covariances.to(device)
        self.weights = weights.to(device)
        # Preparation for posterior computation:
        const = torch.Tensor([-0.5 * self.means.size()[1] * np.log(2 * np.pi)]).to(self.means.device)
        self._inv_covariances = torch.inverse(self.covariances)
        self._component_constants = torch.zeros(self.weights.numel(), device=self.means.device)
        for i in range(self.weights.numel()):
            self._component_constants[i] = -0.5 * torch.logdet(self.covariances[i, :, :]) + const + torch.log(self.weights[i])

    def to_device(self, device):
        return Gmm(self.means, self.covariances, self.weights, device=device)

    def compute_posteriors_top_select(self, frames, top_indices):
        logprob = torch.zeros(top_indices.size(), device=self.means.device)
        for i in range(self.weights.numel()):
            indices_of_component = (top_indices == i)
            frame_selection = torch.any(indices_of_component, 0)
            post_index = torch.argmax(indices_of_component, 0)[frame_selection]
            centered_frames = frames[frame_selection, :] - self.means[i, :]        
            logprob[post_index, frame_selection] = self._component_constants[i] - 0.5 * torch.sum(torch.mm(centered_frames, self._inv_covariances[i, :, :]) * centered_frames, 1)
        llk = torch.logsumexp(logprob, dim=0)
        return torch.exp(logprob - llk)

    def compute_posteriors(self, frames):
        logprob = torch.zeros(self.weights.numel(), frames.size()[0], device=self.means.device)
        for i in range(self.weights.numel()):
            centered_frames = frames - self.means[i, :]
            logprob[i, :] = self._component_constants[i] - 0.5 * torch.sum(torch.mm(centered_frames, self._inv_covariances[i, :, :]) * centered_frames, 1)
        llk = torch.logsumexp(logprob, dim=0)
        return torch.exp(logprob - llk)

    def save_npz(self, filename):
        np.savez(filename, weights=self.weights.cpu().numpy(), means=self.means.cpu().numpy(), covariances=self.covariances.cpu().numpy())
        print('GMM saved to {}'.format(ensure_npz(filename)))
    
    @classmethod
    def from_npz(cls, filename, device):
        data = np.load(ensure_npz(filename))
        weights = torch.from_numpy(data['weights'])
        means = torch.from_numpy(data['means'])
        covariances = torch.from_numpy(data['covariances'])
        return Gmm(means, covariances, weights, device)

    @classmethod
    def from_kaldi(cls, filename, device):
        ubm = KaldiFullGmm()
        with kio.xopen(filename) as f:
            ubm.read(f.stream(), f.binary)
        means = torch.from_numpy(ubm.get_means().numpy())
        weights = torch.from_numpy(ubm.weights().numpy())
        n_components = weights.numel()
        feat_dim = means.size()[1]
        covariances = torch.zeros([n_components, feat_dim, feat_dim], device='cpu', dtype=torch.float32)
        for index, kaldicovar in enumerate(ubm.get_covars()):
            covariances[index, :, :] = torch.from_numpy(KaldiMatrix(kaldicovar).numpy())
        return Gmm(means, covariances, weights, device=device)

class DiagGmm():
    def __init__(self, means, covariances, weights, device=torch.device("cpu")):
        self.means = means.to(device)
        self.covariances = covariances.to(device)
        self.weights = weights.to(device)
        # Preparation for posterior computation:
        const = torch.Tensor([self.means.size()[1] * np.log(2 * np.pi)]).to(self.means.device)
        self.posterior_constant = torch.sum(self.means * self.means / self.covariances, 1) + torch.sum(torch.log(self.covariances), 1) + const
        self.posterior_constant = self.posterior_constant.unsqueeze(1)
        self.precisions = (1 / self.covariances)
        self.mean_pres = (self.means / self.covariances)

    def compute_posteriors(self, frames):
        logprob = torch.mm(self.precisions, (frames * frames).t()) - 2 * torch.mm(self.mean_pres, frames.t())
        logprob = -0.5 * (logprob + self.posterior_constant)
        logprob = logprob + torch.log(self.weights.unsqueeze(1))
        llk = torch.logsumexp(logprob, 0)
        return torch.exp(logprob - llk)

    @classmethod
    def from_full_gmm(cls, full_gmm, device):
        means = full_gmm.means.clone()
        weights = full_gmm.weights.clone()
        covariances = torch.zeros(means.size(), device=full_gmm.covariances.device, dtype=full_gmm.covariances.dtype)
        for index in range(weights.numel()):
            covariances[index, :] = full_gmm.covariances[index, :, :].diag()
        return DiagGmm(means, covariances, weights, device=device)
           
    