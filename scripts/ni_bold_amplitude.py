#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: soojinlee
"""

import os
import numpy as np
from scipy.stats import zscore
import nibabel as nib
from ni_myutils import *


# Path information
basedir = '/vols/Data/ukbiobank/FMRIB/IMAGING/data3/SubjectsAll'

# Load subject IDs
subjid = np.load('subjectid.npz')
subjid = subjid['subjid']
nsubj = len(subjid)

# Load MNI mask
maskmni = nib.load('MNI152_T1_2mm_brain_mask.nii.gz')
maskmni = maskmni.get_fdata()
maskmni = maskmni.flatten('F').astype('bool')

# Load group ICA maps
Gm = nib.load('melodic_IC.nii.gz').get_fdata()
Gm = vol2vec(Gm) # ICs x voxel
Gm = Gm[:, maskmni] # apply mask
nvox = Gm.shape[1]
goodnode = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]  # good ICs (i.e., excluding noise components)
Gm = Gm[goodnode, :]  # (21, 228483)


# Initialize variables
BOLDamp = np.zeros((nsubj, 21))
BOLDamp_weighted = np.zeros((nsubj, 21))


for s, sid in enumerate(subjid):
    fpath = '{}/{}/fMRI/rfMRI.ica/reg_standard/filtered_func_data_clean.nii.gz'.format(basedir, sid)
    # Read fMRI data
    X = nib.load(fpath)
    X = X.get_fdata()
    X = vol2vec(X)  # time x voxel
    
    # Get voxelwise SD of BOLD signals
    voxstd = np.std(X[:, maskmni], axis=0, ddof=1).reshape(1,-1)
    
    # Initialize temporary variable
    tmp_mean = np.zeros(21)
    tmp_wavg = np.zeros(21)
    
    # Get BOLD amplitude for each network
    for i in range(21):
        G = Gm[i,:].reshape(1,-1)

        # compute mean BOLD amplitude across the thresholded voxels
        Gmask = np.abs(G) > 3.29
        tmp_mean[i] = np.nanmean(voxstd[Gmask])
        
        # compute weighted average BOLD amplitude
        tmp_wavg[i] = (voxstd @ G.T)/nvox
    
    BOLDamp[s] = tmp_mean
    BOLDamp_weighted[s] = tmp_wavg
    
    # Intermediate save
    if s % 100 == 0:
        np.savez('bold_amplitude', BOLDamp=BOLDamp, BOLDamp_weighted=BOLDamp_weighted, subjid=subjid)

# Final save
np.savez('bold_amplitude', BOLDamp=BOLDamp, BOLDamp_weighted=BOLDamp_weighted, subjid=subjid)
