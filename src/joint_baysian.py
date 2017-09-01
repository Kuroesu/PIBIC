#coding=utf-8
import sys
import numpy as np
from copy import copy
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.externals import joblib


#ratio of similar,the threshold we always choose in (-1,-2)            
def Verify(A, G, x1, x2):

    ratio = np.dot(np.dot(np.transpose(x1),A),x1) + np.dot(np.dot(np.transpose(x2),A),x2) - 2*np.dot(np.dot(np.transpose(x1),G),x2)
    return ratio

def trainJBC(matchPairs, missMatchPairs):
    print "match",len(matchPairs),"x",len(matchPairs[0])
    print "miss",len(missMatchPairs),"x",len(missMatchPairs[0])
    Xi = matchPairs
    Xe = missMatchPairs
    
    Si = np.cov(Xi,Xi)
    Se = np.cov(Xe,Xe)
    
    print "Si",len(Si),"x",len(Si[0])
    print "Se",len(Se),"x",len(Se[0])
    
    MFG = np.linalg.inv(Si)
    

    MF = copy(MFG)
    MF[:len(MFG)/2,len(MFG)/2:] = [0 for _ in xrange(len(MFG)/2)]
    MF[len(MFG)/2:,:len(MFG)/2] = [0 for _ in xrange(len(MFG)/2)]
    
    MA = np.subtract(np.linalg.inv(Se) , MF)
    
    print "MF",len(MF),"x",len(MF[0])
    print "MA",len(MA),"x",len(MA[0])
    
    A = MA[:len(MA)/2,:len(MA)/2]
    G = MFG[:len(MA)/2,len(MA)/2:]
    
    print "A",len(A),"x",len(A[0])
    print "G",len(G),"x",len(G[0])
    
    return A,G

