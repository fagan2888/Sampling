# -*- coding: utf-8 -*-
"""
Created on Tue May  7 12:00:20 2013

@author: hok1
"""

from OutlierDetection import OutlierDetector

class ConvergedOutlierDetector:
    def __init__(self, vectors, outz=3.):
        self.vectors = vectors
        self.filteredVec = vectors
        self.outz = outz
        self.trainConvergedDetector()
        
    def trainConvergedDetector(self):
        numVecs = len(self.filteredVec)
        converged = False
        i = 0
        while not converged:
            outdetect = OutlierDetector()
            outdetect.importVectors(self.filteredVec)
            insider = lambda vec: outdetect.isOutlier(vec,
                                                      stdThreshold=self.outz)
            self.filteredVec = filter(insider, self.filteredVec)
            if numVecs == len(self.filteredVec):
                converged = True
            numVecs = len(self.filteredVec)
            i += 1
            #print 'Train ', i, '  # vectors = ', len(self.filteredVec)
        
        self.outDetect = OutlierDetector()
        self.outDetect.importVectors(self.filteredVec)
        
    def transformVec(self, vector):
        return self.outDetect.transformVec(vector)
        
    def normalizeVec(self, vector):
        return self.outDetect.normalizeVec(vector)
        
    def isOutlier(self, vector):
        return self.outDetect.isOutlier(vector, stdThreshold=self.outz)
