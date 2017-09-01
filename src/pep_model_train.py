from spatialAppearanceFeature import extract
from sklearn.mixture import GMM 
from numpy import transpose
import pandas as pd
from sklearn.externals import joblib
from joint_baysian import trainJBC
from joint_baysian import Verify

#--------------------------------------------------TRAIN---------------------------------------------

trainingSet = [["Aaron_Peirsol_0001.jpg","Aaron_Peirsol_0002.jpg"],#---match
               ["Aaron_Peirsol_0003.jpg","Aaron_Peirsol_0004.jpg"],
               ["Aaron_Sorkin_0001.jpg","Aaron_Sorkin_0001.jpg"],
               
               ["AJ_Cook_0001.jpg","Marsha_Thomason_0001.jpg"],  #-----miss
               ["Aaron_Sorkin_0002.jpg","Frank_Solich_0005.jpg"],
               ["Abdel_Nasser_Assidi_0002.jpg","Hilary_McKay_0001.jpg"]]


num_training_pairs = 3
pair_cont = 0 
num_components = 10

data = pd.DataFrame()#training set pep-model
matchPair = []#training set joint
missPair = []#training set joint
for pair in trainingSet:
    print pair
    for img in pair:
        path = "train/%s" % img
        bag,l = extract(path)
        data.append(bag)
        if(pair_cont < num_training_pairs):
            for feature in bag.as_matrix():
                matchPair.append(feature[:-2])
        else:
            for feature in bag.as_matrix():
                missPair.append(feature[:-2])
    pair_cont += 1
    
    
# GMM = GMM(n_components=num_components,covariance_type='spherical', n_iter=20)
# GMM.fit(data)      
GMM = joblib.load("gmm_11_10.pkl") 

A,G = trainJBC(transpose(matchPair),transpose(missPair))


#--------------------------------------------------TEST---------------------------------------------

testingSet = [["Abdullah_Gul_0013.jpg","Abdullah_Gul_0014.jpg"],#match pair
               ["AJ_Lamas_0001.jpg","Zach_Safrin_0001.jpg"]] #missmatch pair

num_testing_pair = 1
pair_cont = 0 

match = []
miss = []

for pair in testingSet:
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
              
    if(pair_cont < num_testing_pair):
        match.append([pep_repre1,pep_repre2])
        pair_cont+=1
    else:
        miss.append([pep_repre1,pep_repre2])    
        pair_cont+=1
        
        
for pair in match:
    for i in Verify(A, G, transpose(pair[0]),transpose(pair[1])): 
        print len(i),len(i[0])
print "-------------------------"
for pair in miss:
    for i in Verify(A, G, transpose(pair[0]),transpose(pair[1])): 
        print len(i),len(i[0])
                    
