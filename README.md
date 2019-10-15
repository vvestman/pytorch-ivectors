#### GPU accelerated PyTorch implementation of frame posterior computation and i-vector extractor training.
Kaldi is required for MFCC extraction and UBM training.

#### Steps to run example script with VoxCeleb data:
- Move **kaldi/egs/voxceleb/v1/extract_feats_and_train_ubm.sh** to the corresponding folder in your Kaldi installation
- In **extract_feats_and_train_ubm.sh**, update **output_dir**, **voxceleb1_root**, and **voxceleb2_root**.
  - If you are using newer version of VoxCeleb1 (1.1), you might have to modify **kaldi/egs/voxceleb/v1/local/make_voxceleb1.pl** as the data organization is different than in the original VoxCeleb release.
- run **extract_feats_and_train_ubm.sh**
- update **DATA_FOLDER** in **run_voxceleb_ivector.py**
- install and activate compatible conda environment
  - **environment.yml** has all the needed packages
  - Main requirements: Python (>3.6), PyTorch(>1.1), NumPy, SciPy, PyKaldi
- run **run_voxceleb_ivector.py**


For more details:
http://dx.doi.org/10.21437/Interspeech.2019-1955

```
@inproceedings{Vestman2019,
  author={Ville Vestman and Kong Aik Lee and Tomi H. Kinnunen and Takafumi Koshinaka},
  title={{Unleashing the Unused Potential of i-Vectors Enabled by GPU Acceleration}},
  year=2019,
  booktitle={Proc. Interspeech 2019},
  pages={351--355},
  doi={10.21437/Interspeech.2019-1955},
  url={http://dx.doi.org/10.21437/Interspeech.2019-1955}
}
```

