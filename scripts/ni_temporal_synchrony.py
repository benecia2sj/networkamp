#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: soojinlee
"""

import os
import numpy as np
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
Gm = np.swapaxes(Gm, 0, 1) # voxel x ICs
Gm = Gm - np.mean(Gm, axis=0, keepdims=True)


# Initialize variable for temporal synchrony
TS = np.zeros((nsubj, 21))

for s, sid in enumerate(subjid):
    # Read fMRI data from a subject
    X = nib.load('%s/%d/fMRI/rfMRI.ica/reg_standard/filtered_func_data_clean.nii.gz' % (basedir, sid))
    X = X.get_fdata()
    X = vol2vec(X)
    X = X[:, maskmni]
    X = np.swapaxes(X, 0, 1) # voxel x time
    X = X - np.mean(X, axis=0, keepdims=True)
    
    # Voxelwise standardization (mean=0, SD=1) of BOLD signals 
    Xm = nets_normalize(np.copy(X), dim=1)
    Xm[np.isnan(Xm)] = 0
    
    # Dual Regression Stage 1
    Mnew = np.linalg.pinv(np.matmul(Gm.T, Gm))@Gm.T@Xm
    Mnew = Mnew.T
    Mnew = Mnew - np.mean(Mnew, axis=0, keepdims=True) # remove mean for each node
    ts = make_ts(Mnew)
    ts = nets_tsclean(ts, True)
    TS[s] = np.std(ts['ts'], axis=0, ddof=1)
    
    # Intermediate save
    if s % 100 == 0:
        np.savez('temporal_synchrony', TS=TS, subjid=subjid)

# Final save
np.savez('temporal_synchrony', TS=TS, subjid=subjid)
