# This is an example config file.
# Format is txt, but I use py-extension to get the color highlighting that the editor provides.

# Lines starting with # are comment lines and the config parser ignores them.
# Comment sign # only works if it is placed in the beginning of the line.

# Empty lines separate settings for different runs.
# Different runs will be run one after another by run_voxceleb_xvector_standalone.py.
# After each run the settings will be reverted to their default values.
# That is, settings of the previous run do not affect the settings of the next run.

# The settings that are not specified in this file will have the default values defined in the settings classes.


# RUN 0:
# Compute and save frame posteriors to disk
# This can be commented out after it has been done once.
#recipe.start_stage = 0
#recipe.end_stage = 1

# RUN 1:
# Try two different variations of augmented model training (with and without residualcovariance updates):
recipe.start_stage = 2
ivector.type = ['kaldi']
ivector.update_covariances = [True, False]
ivector.minimum_divergence = [True]
ivector.update_means = [True]
#Increase the number of iterations to improve performance:
ivector.n_iterations = 5
ivector.dataloader_workers = 44
ivector.ivec_dim = 400

# RUN 2:
# Try all four parameter combinations of residual updates (True/False) and minimum_divergence (True/False) with standard formulation:
recipe.start_stage = 2
ivector.type = ['standard']
ivector.update_covariances = [True, False]
ivector.minimum_divergence = [True, False]
ivector.update_means = [False]
#Increase the number of iterations to improve performance:
ivector.n_iterations = 5
ivector.dataloader_workers = 44
ivector.ivec_dim = 400