# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 09:52:06 2013

@author: hok1
"""

import numpy as np
from functools import partial

from VectorStatistics import vecMean, covMat

def MarkovChainByGibbsSampling(distfunc, nSamples, ndim):
    # implementing Gibbs sampling
    startVec = np.array([0.]*ndim)
    normRnd = [[np.random.normal() for j in range(ndim)] for i in range(nSamples)]
    unifRnd = [[np.random.uniform() for j in range(ndim)] for i in range(nSamples)]
    vecArray = np.zeros([nSamples, ndim])
    alphaArray = np.zeros([nSamples, ndim])
    for i in range(nSamples):
        vecArray[i] = startVec if i==0 else vecArray[i-1]
        for j in range(ndim):
            previousVec = vecArray[i-1]
            vecArray[i][j] = normRnd[i][j] + previousVec[j]
            alphaArray[i][j] = min(1., distfunc(vecArray[i])/distfunc(previousVec))
            vecArray[i][j] = vecArray[i][j] if alphaArray[i][j]>unifRnd[i][j] else previousVec[j]
    return vecArray

def unnormalizedMultivariateGaussianDist(vec, meanVec, covMat):
    dVec = np.transpose(np.matrix(vec-meanVec))
    invCovMat = np.linalg.inv(covMat)
    return np.exp(-0.5*np.transpose(dVec)*np.matrix(invCovMat)*dVec)

def testrun():
    #meanVector = np.array([1., -1.])
    #covMatrix = np.array([[1., 0], [0, 2.]])
    meanVector = np.array([1., -1., 0.])
    covMatrix = np.array([[2., 0.1, 0.], [0.1, 1.5, -0.1], [0., -0.1, 0.01]])
    ndim = len(meanVector)

    f = partial(unnormalizedMultivariateGaussianDist,
                meanVec = meanVector,
                covMat = covMatrix)

    chain = MarkovChainByGibbsSampling(f, 100000, ndim)
    
    print vecMean(chain)
    
    print covMat(chain)

if __name__ == '__main__':
    testrun()
