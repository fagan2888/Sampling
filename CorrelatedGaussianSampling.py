# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 10:06:45 2013

@author: hok1
"""

import numpy as np

defaultMeanVector = np.array([0, 0])
defaultCovarianceMatrix = np.array([[1, 0], [0, 1]])

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
        #uValidMatrix = (u == np.conjugate(np.transpose(u)))
        #vValidMatrix = (v == np.conjugate(np.transpose(v)))
        #if not ((False in uValidMatrix) and (False in vValidMatrix)):
        #    raise NotHermiteanMatrixException()
        self.U = v
        self.diagVar = s
        self.invU = np.linalg.inv(self.U)
        
    def sampleOneVector(self):
        sampleVec = np.zeros(self.ndim)
        for i in range(self.ndim):
            sampleVec[i] = np.random.normal(scale=1./np.sqrt(self.diagVar[i]))
        procSampleVec = np.matrix(self.invU) * np.transpose(np.matrix(sampleVec))
        procSampleVec = np.array(procSampleVec.transpose()) + self.meanVector
        return procSampleVec[0]
        
def test():
    numSamples = 15000
    meanVector = np.array([1., -1., 0.])
    covMatrix = np.array([[1., 0.1, 0.], [0.1, 1.5, -0.1], [0., -0.1, 0.01]])
    ndim = len(meanVector)
    sampler = CorrelatedGaussianSampler(meanVector=meanVector,
                                        covMatrix=covMatrix)
    sampledVectors = []
    for i in range(numSamples):
        sampledVector = sampler.sampleOneVector()
        sampledVectors.append(sampledVector)
    
    # find the mean vector    
    sumVec = np.zeros(ndim)
    for i in range(numSamples):
        sumVec += sampledVectors[i]
        samVec = np.matrix(sampledVectors[i])
    avgVector = np.transpose(np.matrix(sumVec/numSamples))
    print avgVector

    # find the covariance matrix
    sumMat = np.matrix(np.zeros([ndim, ndim]))
    for i in range(numSamples):
        samVec = np.transpose(np.matrix(sampledVectors[i]))
        diffVec = samVec - avgVector
        sumMat += diffVec*np.transpose(diffVec)
    sumMat *= 0.5
    for i in range(ndim):
        sumMat[i, i] *= 2
    print sumMat/numSamples
    
if __name__ == '__main__':
    test()
