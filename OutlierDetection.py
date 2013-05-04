# -*- coding: utf-8 -*-
"""
Created on Sat May  4 16:17:25 2013

@author: hok1
"""

from VectorStatistics import vecMean, covMat
import numpy as np
from operator import div

class OutlierDetector:
    def __init__(self):
        self.trainVectors = []
        self.covMatrix = []
    
    def importVectors(self, vectors):
        self.trainVectors = vectors
        self.covMatrix = covMat(vectors)
        self.vecMean = vecMean(vectors)
        
        # PCA
        u, s, v = np.linalg.svd(self.covMatrix)
        self.diagCovVal = s
        self.U = v
        self.invU = np.linalg.inv(v)
        
    def transformVec(self, vector):
        diffvec = vector - self.vecMean
        transVec = np.matrix(self.U)*np.transpose(np.matrix(diffvec))
        return np.array(np.transpose(transVec))[0]
        
    def normalizeVec(self, vector):
        transVec = self.transformVec(vector)
        return map(div, transVec, self.diagCovVal)
        
    def isOutlier(self, vector, stdThreshold=2.):
        normVec = self.normalizeVec(vector)
        if np.linalg.norm(normVec) < stdThreshold:
            return False
        else:
            return True
