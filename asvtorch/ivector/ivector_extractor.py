import time
import datetime

import torch
import numpy as np

import asvtorch.ivector.statloader
from asvtorch.misc.misc import ensure_npz
from asvtorch.ivector.gmm import Gmm

class IVectorExtractor():
    def __init__(self, t_matrix, means, inv_covariances, prior_offset, device):
        # When prior offset is zero, standard (non-augmented) i-vector formulation is used.
        self.t_matrix = t_matrix.to(device)
        self.prior_offset = prior_offset.to(device)
        self.means = means.to(device)
        self.inv_covariances = inv_covariances.to(device)
        self.n_components, self.ivec_dim, self.feat_dim = self.t_matrix.size()
        self.identity = torch.eye(self.ivec_dim, device=device).unsqueeze(0)
        self.bias_offset = None

    def _compute_posterior_means_and_covariances(self, n_all, f_all, batch_size, component_batches):
        covariances = torch.zeros(self.ivec_dim, batch_size, self.ivec_dim, device=self.t_matrix.device)
        means = torch.zeros(self.ivec_dim, batch_size, device=self.t_matrix.device)
        for bstart, bend in component_batches:
            n = n_all[:, bstart:bend]
            f = f_all[bstart:bend, :, :]
            sub_t = self.t_matrix[bstart:bend, :, :]
            sub_inv_covars = self.inv_covariances[bstart:bend, :, :]
            sub_tc = torch.bmm(sub_t, sub_inv_covars)
            tt = torch.bmm(sub_tc, torch.transpose(sub_t, 1, 2))
            tt.transpose_(0, 1)
            covariances += torch.matmul(n, tt)
            means = torch.addbmm(means, sub_tc, f)        
        covariances.transpose_(0, 1)
        covariances += self.identity
        covariances = torch.inverse(covariances)
        means.t_()
        means[:, 0] += self.prior_offset
        means.unsqueeze_(2)
        means = torch.bmm(covariances, means)
        means = means.view((means.size()[:2]))
        return means, covariances

    def _get_component_batches(self, n_component_batches):
        cbatch_size = self.n_components // n_component_batches
        component_batches = []
        for cbatch_index in range(n_component_batches):
            bstart = cbatch_index * cbatch_size
            bend = (cbatch_index + 1) * cbatch_size
            component_batches.append((bstart, bend))
        return component_batches

    def _get_stat_loader(self, rxspecifiers, feature_loader, second_order, batch_size, n_workers):
        data_dims = (self.n_components, self.feat_dim)
        if self.prior_offset == 0:
            stat_loader = asvtorch.ivector.statloader.get_stat_loader(rxspecifiers, feature_loader, data_dims, batch_size, second_order, self.means, n_workers)
        else:  # Kaldi style i-vector (augmented form) --> No centering required
            stat_loader = asvtorch.ivector.statloader.get_stat_loader(rxspecifiers, feature_loader, data_dims, batch_size, second_order, None, n_workers)
        return stat_loader

    def get_updated_ubm(self, ubm, device):
        if self.prior_offset == 0:
            means = self.means.clone()
        else:
            means = self.t_matrix[:, 0, :] * self.prior_offset
        covariances = ubm.covariances.clone()
        weights = ubm.weights.clone()
        return Gmm(means, covariances, weights, device)
 
    def extract(self, rxspecifiers, feature_loader, settings):
        stat_loader = self._get_stat_loader(rxspecifiers, feature_loader, False, settings.batch_size_in_utts, settings.dataloader_workers)
        component_batches = self._get_component_batches(settings.n_component_batches)        
        print('Extracting i-vectors for {} utterances...'.format(len(rxspecifiers[0])))
        start_time = time.time()
        ivectors = torch.zeros(len(rxspecifiers[0]), self.ivec_dim, device=self.t_matrix.device) 
        counter = 0
        for batch_index, batch in enumerate(stat_loader):
            n_all, f_all = batch
            batch_size = n_all.size()[0]
            print('{:.0f} seconds elapsed, Batch {}/{}: utterance count = {}'.format(time.time() - start_time, batch_index+1, stat_loader.__len__(), batch_size))
            n_all = n_all.to(self.t_matrix.device)
            f_all = f_all.to(self.t_matrix.device)
            means = self._compute_posterior_means_and_covariances(n_all, f_all, batch_size, component_batches)[0]
            ivectors[counter:counter+batch_size, :] = means
            counter += batch_size       
        ivectors[:, 0] -= self.prior_offset
        print('I-vector extraction completed in {:.0f} seconds.'.format(time.time() - start_time))
        return ivectors
  
    def train(self, rxspecifiers, feature_loader, output_filename, settings, resume=0):
        if resume < 0:
            resume = 0
        elif resume > 0:
            print('Resuming i-vector extractor training from iteration {}...'.format(resume))
            extractor = IVectorExtractor.from_npz('{}.{}'.format(ensure_npz(output_filename, inverse=True), resume), self.t_matrix.device)
            self.t_matrix = extractor.t_matrix
            self.means = extractor.means
            self.inv_covariances = extractor.inv_covariances
            self.prior_offset = extractor.prior_offset

        print('Training i-vector extractor ({} iterations)...'.format(settings.n_iterations))

        n_utts = len(rxspecifiers[0])
        component_batches = self._get_component_batches(settings.n_component_batches)
        
        print('Allocating memory for accumulators...')
        z = torch.zeros(self.n_components, device=self.t_matrix.device)
        S = torch.zeros(self.n_components, self.feat_dim, self.feat_dim, device=self.t_matrix.device)
        Y = torch.zeros(self.n_components, self.feat_dim, self.ivec_dim, device=self.t_matrix.device)
        R = torch.zeros(self.n_components, self.ivec_dim, self.ivec_dim, device=self.t_matrix.device)  # The biggest memory consumer!
        h = torch.zeros(self.ivec_dim, device=self.t_matrix.device)
        H = torch.zeros(self.ivec_dim, self.ivec_dim, device=self.t_matrix.device)   
        
        iteration_times = []
        start_time = time.time()        
        for iteration in range(1, settings.n_iterations + 1):       
            iter_start_time = time.time()

            print('Initializing statistics loader...')
            accumulate_2nd_stats = settings.update_covariances and iteration == 1  
            stat_loader = self._get_stat_loader(rxspecifiers, feature_loader, accumulate_2nd_stats, settings.batch_size_in_utts, settings.dataloader_workers)        
            
            print('Iterating over batches of utterances...')
            for batch_index, batch in enumerate(stat_loader):            
                
                if accumulate_2nd_stats:
                    n_all, f_all, s_batch_sum = batch
                    S += s_batch_sum.to(self.t_matrix.device)  # 2nd order stats need to be accumulated only once
                else:
                    n_all, f_all = batch
                              
                batch_size = n_all.size()[0]
                print('Iteration {} ({:.0f} seconds), Batch {}/{}: utterance count = {}'.format(iteration + resume, time.time() - iter_start_time, batch_index+1, stat_loader.__len__(), batch_size))

                n_all = n_all.to(self.t_matrix.device)
                f_all = f_all.to(self.t_matrix.device)
                if iteration == 1:  # Need to be accumulated only once
                    z += torch.sum(n_all, dim=0)

                means, covariances = self._compute_posterior_means_and_covariances(n_all, f_all, batch_size, component_batches)
                
                # Accumulating...
                h += torch.sum(means, dim=0)
                yy = torch.baddbmm(covariances, means.unsqueeze(2), means.unsqueeze(1))
                H += torch.sum(yy, dim=0)
                yy = yy.permute((1, 2, 0))                
                for bstart, bend in component_batches: # Batching over components saves GPU memory
                    n = n_all[:, bstart:bend]
                    f = f_all[bstart:bend, :, :]                    
                    Y[bstart:bend, :, :] += torch.matmul(f, means)
                    R[bstart:bend, :, :] += torch.matmul(yy, n).permute((2, 0, 1))

            self.weights = z / torch.sum(z) * n_utts
            h = h / n_utts
            H = H / n_utts
            H = H - torch.ger(h, h)

            # Updating:
            if settings.update_projections: self._update_projections(Y, R, component_batches)
            if settings.update_covariances: self._update_covariances(Y, R, z, S, component_batches)
            if settings.minimum_divergence: self._minimum_divergence_whitening(h, H, component_batches)
            if settings.update_means:       self._minimum_divergence_centering(h, component_batches)
        
            print('Zeroing accumulators...')
            Y.zero_()
            R.zero_()
            h.zero_()
            H.zero_()
            #G.zero_()

            if settings.save_every_iteration:
                self.save_npz('{}.{}'.format(ensure_npz(output_filename, inverse=True), iteration + resume))

            iteration_times.append(time.time() - iter_start_time)

        self.save_npz(output_filename)
        print('Training completed in {:.0f} seconds.'.format(time.time() - start_time))
        return iteration_times
          
    def _update_projections(self, Y, R, component_batches):
        print('Updating projections...')
        for bstart, bend in component_batches:
            self.t_matrix[bstart:bend, :, :] = torch.potrs(Y[bstart:bend, :, :].transpose(1, 2), torch.cholesky(R[bstart:bend, :, :], upper=True))

    def _update_covariances(self, Y, R, z, S, component_batches):
        print('Updating covariances...')
        for bstart, bend in component_batches:
            crossterm = torch.matmul(Y[bstart:bend, :, :], self.t_matrix[bstart:bend, :, :])
            crossterm = crossterm + crossterm.transpose(1, 2)        
            self.inv_covariances[bstart:bend, :, :] = S[bstart:bend, :, :] - 0.5 * crossterm

        var_floor = torch.sum(self.inv_covariances, dim=0)
        var_floor *= 0.1 / torch.sum(z)
        self.inv_covariances = self.inv_covariances / z.unsqueeze(1).unsqueeze(1)
        self._covariances = (self.inv_covariances).clone()
        self._apply_floor_(self.inv_covariances, var_floor, component_batches)
        self.inv_covariances = torch.inverse(self.inv_covariances)

    def _apply_floor_(self, A, B, component_batches):
        # self._apply_floor_scalar_(B, self._max_abs_eig(B) * 1e-4)  # To prevent Cholesky from failing
        L = torch.cholesky(B)
        L_inv = torch.inverse(L)
        num_floored = 0
        batch_size = component_batches[0][1] - component_batches[0][0]
        l = torch.zeros(batch_size, self.feat_dim, device=self.t_matrix.device)
        U = torch.zeros(batch_size, self.feat_dim, self.feat_dim, device=self.t_matrix.device) 
        for bstart, bend in component_batches:
            D = L_inv.matmul(A[bstart:bend, :, :]).matmul(L_inv.t())   
            for c in range(batch_size):
                l[c, :], U[c, :, :] = torch.symeig(D[c, :, :], eigenvectors=True)
            num_floored += torch.sum(l < 1).item()
            l = torch.clamp(l, min=1)
            D = U.matmul(l.unsqueeze(2) * U.transpose(1,2))
            A[bstart:bend, :, :] = L.matmul(D.transpose(1, 2)).matmul(L.t())
        print('Floored {:.1%} of the eigenvalues...'.format(num_floored / (self.n_components * self.feat_dim)))

    def _max_abs_eig(self, A):
        l = torch.symeig(A)[0]
        return torch.max(torch.abs(l))

    def _apply_floor_scalar(self, A, b):
        l, U = torch.symeig(A, eigenvectors=True)
        num_floored = torch.sum(l < b).item()
        l = torch.clamp(l, min=b)
        A = torch.matmul(U, l.unsqueeze(1) * U.t())
        return A, num_floored
        #print('Floored {} eigenvalues of the flooring matrix...'.format(num_floored))

    def _minimum_divergence_whitening(self, h, H, component_batches):
        print('Minimum divergence re-estimation...')
        l, U = torch.symeig(H, eigenvectors=True)
        l = torch.clamp(l, min=1e-7)
        P1 = torch.rsqrt(l) * U  # transposed
        torch.matmul(h, P1, out=h)  # In place operation, so that the result is available for update_means()
        if self.prior_offset != 0:  # Augmented formulation 
            self.prior_offset = h[0]
            print('Prior offset: {}'.format(self.prior_offset))
        P1 = torch.inverse(P1)
        for bstart, bend in component_batches:
            self.t_matrix[bstart:bend, :, :] = P1.matmul(self.t_matrix[bstart:bend, :, :])

    def _minimum_divergence_centering(self, h, component_batches):
        if self.prior_offset == 0:
            self.means += torch.sum(self.t_matrix * h.unsqueeze(0).unsqueeze(2), dim=1)
        else:  # Augmented formulation uses the Householder transformation
            x = h / h.norm()
            alpha = torch.rsqrt(2 * (1 - x[0]))
            print('Alpha: {}'.format(alpha))
            a = x * alpha
            a[0] -= alpha
            P2 = self.identity - 2 * torch.ger(a, a)
            self.prior_offset = torch.dot(h, P2[:, 0].squeeze())
            print('Prior offset: {}'.format(self.prior_offset))
            P2 = torch.inverse(P2)
            for bstart, bend in component_batches:
                self.t_matrix[bstart:bend, :, :] = P2.matmul(self.t_matrix[bstart:bend, :, :])

    def save_npz(self, filename):
        np.savez(filename, t_matrix=self.t_matrix.cpu().numpy(), means=self.means.cpu().numpy(), inv_covariances=self.inv_covariances.cpu().numpy(), prior_offset=self.prior_offset.cpu().numpy())
        print('I-vector extractor saved to {}'.format(ensure_npz(filename)))

    @classmethod
    def random_init(cls, ubm, settings, device, seed=0):
        torch.manual_seed(seed)
        t_matrix = torch.randn(ubm.covariances.size()[0], settings.ivec_dim, ubm.covariances.size()[1])
        means = ubm.means.cpu().clone()
        inv_covariances = ubm._inv_covariances.clone()
        if settings.type == 'kaldi':
            prior_offset = torch.tensor([float(settings.initial_prior_offset)])   
            t_matrix[:, 0, :] = means / prior_offset
        else:
            prior_offset = torch.tensor([float(0)])   
        return IVectorExtractor(t_matrix, means, inv_covariances, prior_offset, device)

    @classmethod
    def from_npz(cls, filename, device, iteration=None):
        if iteration is not None:
            filename = '{}.{}'.format(ensure_npz(filename, inverse=True), iteration)
        data = np.load(ensure_npz(filename))
        t_matrix = torch.from_numpy(data['t_matrix'])
        means = torch.from_numpy(data['means'])
        inv_covariances = torch.from_numpy(data['inv_covariances'])
        prior_offset = torch.from_numpy(data['prior_offset'])
        return IVectorExtractor(t_matrix, means, inv_covariances, prior_offset, device)
