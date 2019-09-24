import pickle

import torch

class VectorProcessor:

    def __init__(self, centering_vectors, whitening_matrices, processing_instruction):
        self.centering_vectors = centering_vectors
        self.whitening_matrices = whitening_matrices
        self.processing_instruction = processing_instruction

    @classmethod
    def train(cls, vectors, processing_instruction, device):
        """[summary]
        
        Arguments:
            vectors {[type]} -- [description]
            processing_instruction {String} -- [Contains characters 'c', 'w', 'l'. For example 'cwlc' performs centering, whitening, length normalization, and centering (2nd time) in this order.]
        """
        print('Training vector processor ...')

        c_count = processing_instruction.count('c')
        w_count = processing_instruction.count('w')

        vec_size = vectors.size()[1]

        whitening_matrices = torch.zeros(w_count, vec_size, vec_size, device=device)
        centering_vectors = torch.zeros(c_count, vec_size, device=device)

        vectors = vectors.to(device)

        c_count = 0
        w_count = 0
        for c in processing_instruction:
            if c == 'c':
                print('Centering...')
                centering_vectors[c_count, :] = torch.mean(vectors, dim=0)
                vectors = vectors - centering_vectors[c_count, :]
                c_count += 1
            elif c == 'w':
                print('Whitening...')
                l, U = torch.symeig(torch.matmul(vectors.t(), vectors) / vectors.size()[0], eigenvectors=True)
                l = torch.clamp(l, min=1e-10)
                whitening_matrices[w_count, :, :] = torch.rsqrt(l) * U  # transposed
                vectors = torch.matmul(vectors, whitening_matrices[w_count, :, :])
                w_count += 1
            elif c == 'l':
                print('Normalizing length...')
                vectors = unit_len_norm(vectors)

        return VectorProcessor(centering_vectors, whitening_matrices, processing_instruction)
            
    def process(self, vectors):
        print('Processing {} vectors ...'.format(vectors.size()[0]))
        vectors = vectors.to(self.centering_vectors.device)
        c_count = 0
        w_count = 0
        for c in self.processing_instruction:
            if c == 'c':
                print('Centering...')
                vectors = vectors - self.centering_vectors[c_count, :]
                c_count += 1
            elif c == 'w':
                print('Whitening...')
                vectors = torch.matmul(vectors, self.whitening_matrices[w_count, :, :])
                w_count += 1
            elif c == 'l':
                print('Normalizing length...')
                vectors = unit_len_norm(vectors)
        return vectors

    def save(self, output_file):
        data = {'c': self.centering_vectors.cpu(), 'w': self.whitening_matrices.cpu(), 'i': self.processing_instruction}
        with open(output_file, 'wb') as outfile:
            pickle.dump(data, outfile, protocol=pickle.HIGHEST_PROTOCOL)
        print('VectorProcessor saved to {}'.format(output_file))

    @classmethod
    def load(cls, input_file, device):
        with open(input_file, 'rb') as infile:
            data = pickle.load(infile)
        centering_vectors = data['c'].to(device)
        whitening_matrices = data['w'].to(device)
        processing_instruction = data['i']
        print('VectorProcessor loaded from {}'.format(input_file))
        return VectorProcessor(centering_vectors, whitening_matrices, processing_instruction)        

def unit_len_norm(data):
    data_norm = torch.sqrt(torch.sum(data ** 2, 1))
    data_norm[data_norm == 0] = 1
    return data / data_norm.unsqueeze(1)
