# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 10:06:45 2013

@author: hok1
"""

from operator import add
import numpy as np

defaultMeanVector = np.array([0, 0])
defaultCovarianceMatrix = np.array([[1, 0], [0, 1]])

vecMean = lambda vecArray: reduce(add, vecArray)/len(vecArray)

def calculateCovMatrix(vecArray):
    avgVec = vecMean(vecArray)
    calDiffVec = lambda vec: (vec-avgVec)
    calMatrix = lambda diffVec: np.transpose(np.matrix(diffVec))*np.matrix(diffVec)
    
    diffVecArray = map(calDiffVec, vecArray)
    covMatArray = map(calMatrix, diffVecArray)
    
    return reduce(add, covMatArray) / len(covMatArray)

class NotHermiteanMatrixException(Exception):
    def __init__(self):
        self.message = 'Not a Hermitran matrix'
        
class NotConsistentSizeOfMatrixException(Exception):
    def __init__(self):
        self.message = 'Matrices in the class inconsistent'

class CorrelatedGaussianSampler:
    def __init__(self, meanVector = defaultMeanVector,
                 covMatrix = defaultCovarianceMatrix):
        self.resetClass(meanVector, covMatrix)
    
    def resetClass(self, meanVector, covMatrix):
        self.meanVector = meanVector
        self.covMatrix = covMatrix
        if (not self.isValidMatrix()):
            raise NotHermiteanMatrixException()
        self.ndim = len(meanVector)
        if not (np.shape(self.covMatrix) == (self.ndim, self.ndim)):
            raise NotConsistentSizeOfMatrixException()
        self.invCovMatrix = np.linalg.inv(covMatrix)
        self.performSVD()        
    
    def isValidMatrix(self):
        equalMatrix = (self.covMatrix == np.conjugate(np.transpose(self.covMatrix)))
        return not (False in equalMatrix)
        
    def performSVD(self):
        u, s, v = np.linalg.svd(self.invCovMatrix)
        self.diagVar = s
        self.U = v
        self.invU = np.linalg.inv(self.U)
        
    def sampleOneVector(self):
        normap = lambda std: np.random.normal(scale=1./np.sqrt(std))
        sampleVec = map(normap, self.diagVar)
        procSampleVec = np.matrix(self.invU) * np.transpose(np.matrix(sampleVec))
        procSampleVec = np.array(procSampleVec.transpose()) + self.meanVector
        return procSampleVec[0]
        
def test():
    numSamples = 15000
    meanVector = np.array([1., -1., 0.])
    covMatrix = np.array([[2., 0.1, 0.], [0.1, 1.5, -0.1], [0., -0.1, 0.01]])
    sampler = CorrelatedGaussianSampler(meanVector=meanVector,
                                        covMatrix=covMatrix)
    sampledVectors = []
    for i in range(numSamples):
        sampledVector = sampler.sampleOneVector()
        sampledVectors.append(sampledVector)
    
    # find the mean vector
    print vecMean(sampledVectors)

    # find the covariance matrix
    print calculateCovMatrix(sampledVectors)
    
if __name__ == '__main__':
    test()
