from sklearn.mixture import GMM
# from BDconnection import getData
from spatialAppearanceFeature import extract
from os import listdir
from time import time
from numba import jit
import pandas as pd
import numpy as np
import cv2
from networkx.algorithms import cluster


    
data = pd.DataFrame()
imgNames = listdir("train")
t = time()
for name in imgNames:
    path = "train/%s" % name
    dataImg = extract(path)
    print(name)
    print('-'*50)
    data = data.append(dataImg)
t2 = time()
print("tempoTotal",(t2-t)) 
num_components = 15
GMM = GMM(n_components=num_components,covariance_type='spherical', n_iter=20)
t1 = time() 
GMM.fit(data)
t2 = time()
print "time fit",(t2-t1)
 
featuresImg1 = extract("George_W_Bush_0179.jpg")
featuresImg2 = extract("George_W_Bush_0193.jpg")
img1M = featuresImg1.as_matrix()
img2M = featuresImg2.as_matrix() 
#lista com o indice do componete gausiano correspondente ao exemplo
gaussianIndex1 = GMM.predict(featuresImg1) 
gaussianIndex2 = GMM.predict(featuresImg2)
#cada linha um exemplo, cada coluna a probabilidades correspondente ao k componente gausiano
predictProbImg1 = GMM.predict_proba(featuresImg1)
predictProbImg2P = GMM.predict_proba(featuresImg2)

pep_repre = [[] for _ in xrange(num_components)]
pep_repreProb= [0 for _ in xrange(num_components)]

pep_repre1 = [[] for _ in xrange(num_components)]
pep_repreProb1= [0 for _ in xrange(num_components)]

for exemple_i in xrange(len(featuresImg1)):
        k = gaussianIndex1[exemple_i]# qual componete exemple_i foi classificado
        # seleciona a maior probabilidade para de cada componente
        if(predictProbImg1[exemple_i][k] > pep_repreProb[k]):
            pep_repreProb[k] = predictProbImg1[exemple_i][k]
            pep_repre[k] = featuresImg1.as_matrix()[exemple_i][:-2]
            
for exemple_i in xrange(len(featuresImg2)):
        k = gaussianIndex2[exemple_i]# qual componete exemple_i foi classificado
        # seleciona a maior probabilidade para de cada componente
        if(predictProbImg1[exemple_i][k] > pep_repreProb1[k]):
            pep_repreProb1[k] = predictProbImg1[exemple_i][k]
            pep_repre1[k] = featuresImg2.as_matrix()[exemple_i][:-2]            
            
                   
        










