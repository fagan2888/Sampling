# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 11:50:36 2013

@author: hok1
"""

from operator import add
import numpy as np

vecMean = lambda vecArray: reduce(add, vecArray)/len(vecArray)

def covMat(vecArray):
    avgVec = vecMean(vecArray)
    calDiffVec = lambda vec: (vec-avgVec)
    calMatrix = lambda diffVec: np.transpose(np.matrix(diffVec))*np.matrix(diffVec)
    
    diffVecArray = map(calDiffVec, vecArray)
    covMatArray = map(calMatrix, diffVecArray)
    
    return reduce(add, covMatArray) / len(covMatArray)
