from collections import defaultdict
import time

import numpy as np
import torch
from scipy.linalg import inv, svd

# Based on the PLDA in LRE 2017 baseline

class Plda:
    def __init__(self, St, Sb):      
        self.St = St
        self.Sb = Sb
        self.plda_dim = 0
        self.l = None
        self.uk = None
        self.qhat = None
    
    @classmethod
    def train_closed_form(cls, data, speaker_labels, device):
        print('Training PLDA...')
        data = data.to(device)
        data, class_boundaries = _rearrange_data(data, speaker_labels)
        print('Computing within class covariance...')
        Sw = _compute_within_cov(data, class_boundaries)
        print('Computing data covariance...')
        St = _compute_cov(data)
        Sb = St - Sw
        print('PLDA trained!...')
        return Plda(St, Sb)

    @classmethod
    def train_em(cls, data, speaker_labels, plda_dim, iterations, device):
        print('Initializing simplified PLDA...')
        data = data.to(device)     
        n_total_sessions, data_dim = data.size()
        F = torch.randn(data_dim, plda_dim, device=device)
        F = _orthogonalize_columns(F)
        S = 1000 * torch.randn(data_dim, data_dim, device=device)
        data_covariance = torch.matmul(data.t(), data)
        data_list, count_list = _arrange_data_by_counts(data, speaker_labels)
        eye_matrix = torch.eye(plda_dim, device=device)

        for iteration in range(1, iterations+1):
            print('Iteration {}...'.format(iteration), end='')
            iter_start_time = time.time()
            
            FS = torch.solve(F, S.t())[0].t()
            FSF = torch.matmul(FS, F) 
                
            dataEh = torch.zeros(data_dim, plda_dim, device=device)
            Ehh = torch.zeros(plda_dim, plda_dim, device=device)
            #print(count_list)
            for count_data, count in zip(data_list, count_list):
                Sigma = torch.inverse(eye_matrix + count * FSF)
                my = torch.chain_matmul(Sigma, FS.repeat(1, count), count_data.view(-1, data_dim * count).t())     
                #print(torch.norm(my[:, 0]))          
                dataEh += torch.matmul(count_data.t(), my.repeat(count, 1).t().reshape(count_data.size()[0], -1))
                Ehh += count * (my.size()[1] * Sigma + torch.matmul(my, my.t()))              
            
            F = torch.solve(dataEh.t(), Ehh.t())[0].t()
            S = (data_covariance - torch.chain_matmul(F, Ehh, F.t())) / n_total_sessions

            Sb = torch.matmul(F, F.t())
            St = Sb + S

            print(' [elapsed time = {:0.1f} s]'.format(time.time() - iter_start_time))
            yield Plda(St, Sb)

    def _compute_scoring_matrices(self, plda_dim):
        if self.plda_dim != plda_dim:
            self.plda_dim = plda_dim
            iSt = torch.inverse(self.St)
            iS = torch.inverse(self.St - torch.chain_matmul(self.Sb, iSt, self.Sb))
            Q = iSt - iS
            P = torch.chain_matmul(iSt, self.Sb, iS)
            U, s = torch.svd(P)[:2]
            self.l = s[:plda_dim]
            self.uk = U[:, :plda_dim]
            self.qhat = torch.chain_matmul(self.uk.t(), Q, self.uk)

    def score_trials(self, model_iv, test_iv, plda_dim):
        self._compute_scoring_matrices(plda_dim)
        model_iv = model_iv.to(self.uk.device)
        test_iv = test_iv.to(self.uk.device)
        model_iv = torch.matmul(model_iv, self.uk)
        test_iv  = torch.matmul(test_iv, self.uk)
        score_h1 = torch.sum(torch.matmul(model_iv, self.qhat) * model_iv, 1)
        score_h2 = torch.sum(torch.matmul(test_iv, self.qhat) * test_iv, 1)
        score_h1h2 = 2 * torch.sum(model_iv * self.l * test_iv, 1)
        scores = score_h1h2 + score_h1 + score_h2
        return scores.cpu().numpy()

    def compress(self, vectors, plda_dim):
        self._compute_scoring_matrices(plda_dim)
        return torch.matmul(vectors, self.uk.to(vectors.device))

    def save(self, filename):
        print('Saving PLDA to file {}'.format(filename))
        np.savez(filename, St=self.St.cpu().numpy(), Sb=self.Sb.cpu().numpy())

    @classmethod
    def load(cls, filename, device):
        print('Loading PLDA from file {}'.format(filename))
        holder = np.load(filename)
        St, Sb = holder['St'], holder['Sb']
        return Plda(torch.from_numpy(St).to(device), torch.from_numpy(Sb).to(device))


def _compute_cov(data):
    data -= torch.mean(data, dim=0)
    cov = torch.matmul(data.t(), data) / (data.size()[0] - 1)
    return cov

def _compute_within_cov(data, class_boundaries):
    data = data.clone()
    for start, end in zip(class_boundaries[:-1], class_boundaries[1:]):
        data[start:end, :] -= data[start:end, :].mean(dim=0)        
    return _compute_cov(data)      

def _rearrange_data(data, speaker_labels):
        print('Rearranging data for PLDA training...')
        index_dict = defaultdict(list)
        for index, label in enumerate(speaker_labels):
            index_dict[label].append(index)
        new_data = torch.zeros(*data.size())
        class_boundaries = [0]
        counter = 0
        for key in index_dict:
            indices = index_dict[key]
            new_data[counter:counter + len(indices), :] = data[indices, :]
            counter += len(indices)
            class_boundaries.append(counter)
        return new_data, class_boundaries

def _orthogonalize_columns(matrix):
    matrix -= torch.mean(matrix, 1).unsqueeze(1)
    D, V = torch.svd(matrix)[1:]
    W = torch.matmul(V, torch.diag((1./(torch.sqrt(D) + 1e-10))))
    return torch.matmul(matrix, W)

def _arrange_data_by_counts(data, labels):
    spk2indices = defaultdict(list)
    for index, label in enumerate(labels):
        spk2indices[label].append(index)

    count2spks = defaultdict(list)
    for spk in spk2indices:
        count2spks[len(spk2indices[spk])].append(spk)

    data_list = []
    count_list = []
    for count in count2spks:
        count_list.append(count)
        count_indices = []
        for spk in count2spks[count]:
            count_indices.extend(spk2indices[spk])
        data_list.append(data[count_indices, :])

    return data_list, count_list