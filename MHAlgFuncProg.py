# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 09:52:06 2013

@author: hok1
"""

import numpy as np
from functools import partial

def MarkovChain(distfunc, nSamples):
    # implementing Metropolis-Hasting Algorithm
    startX = 5.
    normRnd = np.random.normal(size=nSamples)
    unifRnd = np.random.uniform(size=nSamples)
    xArray = np.zeros(nSamples)
    alphaArray = np.zeros(nSamples)
    for i in range(nSamples):
        xArray[i] = startX if i==0 else normRnd[i]+xArray[i-1]
        try:
            alphaArray[i] = min(1., distfunc(xArray[i])/distfunc(startX if i==0 else xArray[i-1]))
        except ZeroDivisionError:
            alphaArray[i] = 1.
        xArray[i] = xArray[i] if alphaArray[i]>unifRnd[i] else (startX if i==0 else xArray[i-1])
    return xArray

def testrun():
    fermidist = lambda x, epsilon, beta: 1/(np.exp(beta*(x-epsilon))+1) if x>=0 else 0
    chain = MarkovChain(partial(fermidist, epsilon=5, beta=10), 15000)
    return chain

if __name__ == '__main__':
    testrun()
