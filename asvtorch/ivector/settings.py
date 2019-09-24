
class PosteriorExtractionSettings():
    def __init__(self):
        # general settings
        self.n_top_gaussians = 20
        self.posterior_threshold = 0.025

        # data loading & batching settings
        self.batch_size_in_frames = 500000
        self.dataloader_workers = 4
        
    def print_settings(self):
        print('POSTERIOR EXTRACTION SETTINGS')
        print('- Number of top Gaussians to select for each frame: {}'.format(self.n_top_gaussians))
        print('- Select Gaussians only if frame posterior is higher than: {}'.format(self.posterior_threshold))
        
        print('Data loading & batching settings')
        print('- Number of data loader workers: {}'.format(self.dataloader_workers))
        print('- Number of frames in a batch: {}'.format(self.batch_size_in_frames))
        print('')



class IVectorSettings():
    
    def __init__(self):

        # general settings
        self.ivec_dim = 400
        self.type = 'kaldi' # 'standard'
        
        # training settings     
        self.n_iterations = 5
        self.initial_prior_offset = 100  # Only useful in the augmented formulation ('kaldi')
        self.update_projections = True
        self.update_covariances = True
        self.minimum_divergence = True
        self.update_means = True
        
        # data loading & batching settings
        self.dataloader_workers = 6
        self.batch_size_in_utts = 200   # Higher batch size will have higher GPU memory usage.
        self.n_component_batches = 16   # must be a power of two! The higher the value, the less GPU memory will be used.
        
        # model saving settings
        self.save_every_iteration = True
        
    
    def print_settings(self):
        print('I-VECTOR EXTRACTOR SETTINGS')
        print('- I-vector type: {}'.format(self.type))
        print('- I-vector dimensionality: {}'.format(self.ivec_dim))
        
        print('Training settings')
        print('- Number of iterations: {}'.format(self.n_iterations))
        if self.type == 'kaldi':
            print('- Initial prior offset: {}'.format(self.initial_prior_offset))
        print('- Update projections (T matrix): {}'.format(self.update_projections))
        print('- Update residual covariances: {}'.format(self.update_covariances))
        print('- Minimum divergence re-estimation: {}'.format(self.minimum_divergence))
        print('- Update means (bias term): {}'.format(self.update_means))
        
        print('Data loading & batching settings')
        print('- Number of data loader workers: {}'.format(self.dataloader_workers))
        print('- Number of utterances in a batch: {}'.format(self.batch_size_in_utts))
        print('- Number of batches for components (has to be power of 2): {}'.format(self.n_component_batches))
        
        print('Saving settings')
        print('- Save model after every iteration: {}'.format(self.save_every_iteration))
        print('')
