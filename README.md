### GPU accelerated PyTorch implementation of frame posterior computation and i-vector extractor training.

#### Steps to run example script with VoxCeleb data:
- Move **kaldi/egs/voxceleb/v1/extract_feats_and_train_ubm.sh** to the corresponding folder in your Kaldi installation
- In **extract_feats_and_train_ubm.sh**, update **output_dir**, **voxceleb1_root**, and **voxceleb2_root**.
  - If you are using newer version of VoxCeleb1 (1.1), you might have to modify **kaldi/egs/voxceleb/v1/local/make_voxceleb1.pl** as the data organization is different than originally.
- run **extract_feats_and_train_ubm.sh**
- update **DATA_FOLDER** in **run_voxceleb_ivector.py**
- install and activate suitable conda environment
  - 
- run **run_voxceleb_ivector.py**

