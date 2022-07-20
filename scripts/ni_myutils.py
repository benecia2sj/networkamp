#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: soojinlee
"""

import numpy as np
from scipy.stats import pearsonr
from scipy.interpolate import interpn


def nets_demean(x, dim=0):
    """
    Removes the average or mean value
    dim = 0: along the column
    dim = 1: along the row
    """
    xdm = x - np.nanmean(x, axis=dim, keepdims=True)
    return xdm


def nets_svds(x, n):
    """
    x is any 2D matrix, which will be approximated by: x=usv'
    n is the number of components to estimate
    if n <=0, we will estimate rank-abs(n) components
    """
    if n < 1:
        n = np.max(np.min(x.shape)+n, 1)
    
    if x.shape[0] < x.shape[1]:
        u = x@x.T
        u[np.isnan(u)] = 0
        u[np.isinf(u)] = 0
        if n < x.shape[0]:
            # return n largest eigenvalues and eigenvectors
            [d, w] = eigsort(u, descend=True)
            d = d[:, :n]
            w = w[:, :n]
        else:
            [d, w] = eigsort(u)
            w = np.fliplr(w)
            d = np.flipud(np.fliplr(d))
        s = np.sqrt(np.abs(d))
        v = x.T@(u@(np.diag(1/np.diag(s))))
    else:
        v = x.T@x
        v[np.isnan(v)] = 0
        v[np.isinf(v)] = 0
        if n < x.shape[1]:
            # return n largest eigenvalues and eigenvectors
            [d, w] = eigsort(v, descend=True)  # d=eigenvalue, w=eigenvectors
            d = d[:, :n]
            w = w[:, :n]
        else:
            [d, w] = eigsort(v)
            w = np.fliplr(w)
            d = np.flipud(np.fliplr(d))            
        s = np.sqrt(np.abs(d))
        u = x@(w@(np.diag(1/np.diag(s))))
    return u, s, v


def nets_unconfound(y, conf, demean=True):
    """
    nets_uncoufound(y, conf)
    nets_unconfound(y, conf, demean=False)
    Regresses conf out of y, handling missing data
    data, confounds and output are all demeaned unless the demean=False is included
    """
    if demean:
        y = nets_demean(y)
        conf = nets_demean(conf)
    
    if np.sum(np.isnan(y.flatten(order='F'))) + np.sum(np.isnan(conf.flatten(order='F'))) == 0:
        conf, _, _ = nets_svds(conf, np.linalg.matrix_rank(conf))
        beta = np.linalg.pinv(conf)@y
        beta[np.abs(beta) < 1e-10] = 0
        yd = y - conf@beta
        if demean:
            yd = nets_demean(yd)
    else:
        r, c = y.shape
        yd = np.zeros((r, c))/0
        for i in range(0, c):
            grot = ~np.isnan(np.sum(np.concatenate((y[:,i].reshape(-1,1), conf), axis=1), axis=1))
            grotconf = conf[grot, :]
            if demean:
                grotconf = nets_demean(grotconf)
            
            grotconf, _, _ = nets_svds(grotconf, np.linalg.matrix_rank(grotconf))
            beta = np.linalg.pinv(grotconf)@y[grot, i]
            beta[np.abs(beta) < 1e-10] = 0
            yd[grot, i] = y[grot, i] - grotconf@beta
            if demean:
                yd[grot, i] = nets_demean(yd[grot, i])
    return yd
    

def nets_normalize(X, dim=1):
    '''
    Remove the Average or mean value and makes the std=1
    along the DIMENSION of X
    X = voxel x time
    dim = 1 (normalize along the time)
    '''
    #remove mean
    m = np.nanmean(X, axis=dim, keepdims=True)
    X = X - m
    
    # devide by standard deviation
    s = np.nanstd(X, axis=dim, ddof=1, keepdims=True)
    X = X / s
    
    X[np.isinf(X)] = 0
    return X


def make_ts(data):
    """
    Make a dictionary variable for nets_netmats module
    Input
    data = time x node
    """
    # By default, the data is assumed to come from ICA25
    goodnodes = [0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
    ts = {'ts':data, 'tr':0.735, 'Nsubjects':1, 'Nnodes':data.shape[1],
          'NnodesOrig':data.shape[1], 'Ntimepoints':data.shape[0],
          'NtimepointsPerSubject':data.shape[0],
          'DD':goodnodes, 'UNK':[]}
    return ts


def nets_tsclean(ts, aggressive=True):
    data = ts['ts']
    NnodesOrig = ts['Nnodes']
    Nnodes = ts['Nnodes']
    DD = ts['DD']
    Ntimepoints = ts['Ntimepoints']
    UNK = ts['UNK']
    
    nongood = np.setdiff1d(np.arange(Nnodes), DD)
    bad = np.setdiff1d(nongood, UNK)
    
    goodTS = data[:, DD]
    badTS = data[:, bad]
    if aggressive:
        newdata = goodTS-badTS@(np.linalg.pinv(badTS)@goodTS)
    else:
        newdata = goodTS
    
    ts['ts'] = newdata
    return ts

    
def nets_netmats(ts, do_rtoz, rho):
    '''
    We only cover 'ridgep' partial correlation using L2-norm Ridge Regression
    as it was used for preprocessing UKB data
    '''
    grot = ts['ts']
    NnodesOrig = ts['Nnodes']
    Nnodes = ts['Nnodes']
    N = grot.shape[1]
    DD = ts['DD']
    Ntimepoints = ts['Ntimepoints']
    UNK = ts['UNK']
    
    grot = np.copy(ts['ts'])
    grot = np.cov(grot.T)
    grot = grot/np.sqrt(np.mean(np.diag(grot)**2))
    grot = np.linalg.inv(grot + rho*np.eye(N))
    
    # ouput_precision==0
    grot = -grot
    tmp =  np.reshape(np.sqrt(np.abs(np.diag(grot))),(1,-1))
    tmp1 = np.tile(tmp.T, (1,N))
    tmp2 = np.tile(tmp, (N,1))
    grot = grot / tmp1 / tmp2
    np.fill_diagonal(grot,0)
    
    # just_diag = 0
    netmats = np.reshape(grot, (1, N*N))
    netmats = 0.5*np.log((1+netmats)/(1-netmats))*(-do_rtoz)
    return netmats


def taketriu(X):
    n = X.shape[0]
    ind = np.triu(np.ones(n),1).astype('bool').flatten(order='F')
    X = np.reshape(X, (1,-1), order='F')
    v = X[0,ind]
    return v


def vol2vec(x):
    '''
    Change volumetric data to matrix
    input: nx x ny x nz x time
    output: time x (nx*ny*nz)
    '''
    x = np.moveaxis(x, -1, 0)
    x = np.reshape(x, (x.shape[0], -1), order='F') # reshape columnwise
    return x


def vec2vol(x, param):
    '''
    Change volumetric data to matrix
    input: time x (nx*ny*nz)
    output: nx x ny x nz x time
    '''
    nx, ny, nz = param['dim']
    nt = x.shape[0]
    nv = nx*ny*nz
    # reconstruct to volume data
    vol = np.zeros((nt, nx, ny, nz))
    for i in range(nt):
        v = np.zeros(nv)
        v[param['mask']] = x[i]
        vol[i] = np.reshape(v, (nx,ny,nz), order='F') # nt, nx, ny, nz     
    vol = np.moveaxis(vol, 0, -1)
    return vol


def vec2netmat(x, nnode=21):
    A = np.triu(np.ones(nnode),1).astype('bool') # upper triangle with 1, the rest is zero
    B = np.zeros((nnode, nnode))
    B.T[A.T] = x
    C = B.T + B
    return C

