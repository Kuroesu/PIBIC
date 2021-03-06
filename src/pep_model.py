from sklearn.mixture import GMM
from sklearn.externals import joblib
# from BDconnection import getData
from spatialAppearanceFeature import extract
from joint_baysian import trainJBC
from os import listdir
from time import time
from numba import jit
import pandas as pd
import numpy as np
import cv2



#     
# data = pd.DataFrame()
# imgNames = listdir("train")
# t = time()
# for name in imgNames:
#     path = "train/%s" % name
#     dataImg,l = extract(path)
#     print(name)
#     print('-'*50)
#     data = data.append(dataImg)
# t2 = time()
# print("tempoTotal",(t2-t)) 


num_components = 10
# GMM = GMM(n_components=num_components,covariance_type='spherical', n_iter=20)
t1 = time() 
# GMM.fit(data)
# joblib.dump(GMM,"gmm_11_10.pkl")
GMM = joblib.load("gmm_11_10.pkl")

t2 = time()
print "time fit",(t2-t1)

Pair = [] 
n = 1
trainN = 3
matchPairs = []
missmatchPairs = []
num_pair_test = 0
num_pair_train = 0

testingSet = [["Abdullah_Gul_0013.jpg","Abdullah_Gul_0014.jpg"],#match pair
               ["AJ_Lamas_0001.jpg","Zach_Safrin_0001.jpg"]] #missmatch pair

trainingSet = [["Aaron_Peirsol_0001.jpg","Aaron_Peirsol_0002.jpg"],#match
               ["Aaron_Peirsol_0003.jpg","Aaron_Peirsol_0004.jpg"],
               ["Aaron_Sorkin_0001.jpg","Aaron_Sorkin_0001.jpg"],
               
               ["AJ_Cook_0001.jpg","Marsha_Thomason_0001.jpg"],#miss
               ["Aaron_Sorkin_0002.jpg","Frank_Solich_0005.jpg"],
               ["Abdel_Nasser_Assidi_0002.jpg","Hilary_McKay_0001.jpg"]]

for pair in trainingSet:
    set_pair = []
    path1 = "test/%s" % pair[0]
    path2 = "test/%s" % pair[1]  
        
    print "extraindo caracteristicas"    
    featuresImg1,l = extract(path1)
    featuresImg2,l = extract(path2)
    
    print "classificando"
    #lista com o indice do componete gausiano correspondente ao exemplo
    predict1 = GMM.predict(featuresImg1) 
    predict2 = GMM.predict(featuresImg2)
    #cada linha um exemplo, cada coluna a probabilidades correspondente ao k componente gausiano
    predictProb1 = GMM.predict_proba(featuresImg1)
    predictProb2 = GMM.predict_proba(featuresImg2)
        
    pep_repre1 = [[] for _ in xrange(num_components)]
    pep_repreProb1= [0 for _ in xrange(num_components)]
    countClass1 = [0 for _ in xrange(num_components)]
    
    pep_repre2 = [[] for _ in xrange(num_components)]
    pep_repreProb2= [0 for _ in xrange(num_components)]
    countClass2 = [0 for _ in xrange(num_components)]
    
    for exemple_i in xrange(len(featuresImg1)):
            k = predict1[exemple_i]# qual componete exemple_i foi classificado
            # seleciona a maior probabilidade para de cada componente
            if(predictProb1[exemple_i][k] > pep_repreProb1[k]):
                countClass1[k]+=1
                pep_repreProb1[k] = predictProb1[exemple_i][k]
                pep_repre1[k] = featuresImg1.as_matrix()[exemple_i][:-2]
                 
                
    for exemple_i in xrange(len(featuresImg2)):
            k = predict2[exemple_i]# qual componete exemple_i foi classificado
            # seleciona a maior probabilidade para de cada componente
            if(predictProb2[exemple_i][k] > pep_repreProb2[k]):
                countClass2[k]+=1
                pep_repreProb2[k] = predictProb2[exemple_i][k]
                pep_repre2[k] = featuresImg2.as_matrix()[exemple_i][:-2]            
    
        
    set_pair+=pep_repre1
    set_pair+=pep_repre2
    if(num_pair_test < n):
        matchPairs+=set_pair
        num_pair_test+=1
    else:
        missmatchPairs+=set_pair
        num_pair_test+=1    
                



trainJBC(matchPairs,missmatchPairs)
                   
        










