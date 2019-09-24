#!/bin/bash
# Copyright   2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#             2017   Johns Hopkins University (Author: Daniel Povey)
#        2017-2018   David Snyder
#             2018   Ewald Enzinger
#
#             2019   Ville Vestman
# Apache 2.0.


. ./cmd.sh
. ./path.sh
set -e

# This script should be run from egs/voxceleb/v1/ folder of your Kaldi installation.
# This script extracts MFCCs and trains the UBM following the original VoxCeleb v1 recipe.

# CHANGE THE FOLLOWING THREE FOLDERS BEFORE RUNNING THE SCRIPT:
output_dir=/media/hdd2/vvestman/voxceleb_outputs
voxceleb1_root=/media/hdd3/voxceleb
voxceleb2_root=/media/hdd3/voxceleb2

mfccdir=$output_dir/mfcc
vaddir=$output_dir/mfcc

stage=0

if [ $stage -le 0 ]; then
  local/make_voxceleb2.pl $voxceleb2_root dev $output_dir/voxceleb2_train
  local/make_voxceleb2.pl $voxceleb2_root test $output_dir/voxceleb2_test
  local/make_voxceleb1.pl $voxceleb1_root $output_dir #IF YOU ARE USING THE NEWEST VERSION OF VOXCELEB1, THIS SCRIPT PROBABLY DOES NOT WORK (data organization changed from the original version)
  
  # We'll train on all of VoxCeleb2, plus the training portion of VoxCeleb1.
  utils/combine_data.sh $output_dir/train $output_dir/voxceleb2_train $output_dir/voxceleb2_test $output_dir/voxceleb1_train
fi

if [ $stage -le 1 ]; then
  # Make MFCCs and compute the energy-based VAD for each dataset
  for name in train voxceleb1_test; do
    steps/make_mfcc.sh --write-utt2num-frames true \
      --mfcc-config conf/mfcc.conf --nj 16 --cmd "$train_cmd" \
      $output_dir/${name} $output_dir/make_mfcc $mfccdir
    utils/fix_data_dir.sh $output_dir/${name}
    sid/compute_vad_decision.sh --nj 16 --cmd "$train_cmd" \
      $output_dir/${name} $output_dir/make_vad $vaddir
    utils/fix_data_dir.sh $output_dir/${name}
  done
fi

if [ $stage -le 2 ]; then
  # Train the UBM.
  sid/train_diag_ubm.sh --cmd "$train_cmd --mem 20G" \
    --nj 16 --num-threads 8 \
    $output_dir/train 2048 \
    $output_dir/diag_ubm

  sid/train_full_ubm.sh --cmd "$train_cmd --mem 40G" \
    --nj 16 --remove-low-count-gaussians false \
    $output_dir/train \
    $output_dir/diag_ubm $output_dir/full_ubm
fi

echo "Done!"
