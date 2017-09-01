import cv2
import pandas as pd
import numpy as np
from time import time
from os import listdir
from sklearn.mixture import GMM
from spatialAppearanceFeature import crop_center
from spatialAppearanceFeature import extract
from spatialAppearanceFeature import gaussianPyramid
from copy import copy
from sklearn.externals import joblib

def getAppearance(coord,win_x,win_y,img):
    linha = coord[0] - ((win_x-1)/2.0)
    coluna = coord[1] - ((win_y-1)/2.0)
    a = img[int(linha):int(linha+win_x),int(coluna):int(coluna+win_y)]    
    return{"a":a,"l":[linha,coluna]}

def getLocal(coord):
    linha = coord[0] - ((win_x-1)/2.0)
    coluna = coord[1] - ((win_y-1)/2.0)
    return [linha,coluna]    
    
#-------------------------------------EXTRACAO-DE-CARACTERISTICAS-----------------

# data = pd.DataFrame()
# imgNames = listdir("train")
# t = time()
# for name in imgNames:
#     path = "train/%s" % name
#     dataImg,l = extract(path)
#     data = data.append(dataImg)
#     print(name)
#     print('-'*50)
# t2 = time()
# print("tempoTotal",(t2-t))

print "-------START TRAIN-------"
  

#------------------------------------------------------PARAMETROS----------------

win_x = 8
win_y = 8    
K = 10

#----------------------------------------------------------TREINO-----------------

# GMM = GMM(n_components=K,covariance_type='spherical', n_iter=20)
t1 = time() 
# GMM.fit(data)

GMM = joblib.load("gmm_11_10.pkl")
# joblib.dump(GMM,"gmm_11pares_1024k.pkl")
t2 = time()
print "time fit",(t2-t1)

#--------------------------------------------------------CLASSIFICACAO-----------------

print "-------START CLASSIFICATION-------"

featureBag,localFeatureBagM = extract("George_W_Bush_0179.jpg")# features  
  
fitProba = GMM.predict_proba(featureBag) #probabilidades de cada exemplo para cada componente
fit = GMM.predict(featureBag)#classificacao do cojunto de treino  


#---------------------------------------------------ESTRUTURAS-------------------------
print "--------START AUX STRUCTS---------"

img = cv2.imread("George_W_Bush_0193.jpg")
img = crop_center(img,150,150)
pep_r = copy(img)
pep_sum = [[[0,0,0] for _ in xrange(img.shape[0])] for _ in xrange(img.shape[0])]
countMatrix = [[ 0 for _ in xrange(img.shape[0])] for _ in xrange(img.shape[0])]

repre_patchs = [{"a":None,"l":[0,0]} for _ in xrange(K)]#guarda o pedaco da img e o local do patch que representa o componente k
max_proba_patchs = [0 for _ in xrange(K)]#guarda as maiores probas para cada k component


#---------------------------------------------------SELECAO-DE-REPRESENTANTES----------------------

print "-------SELECT REPRESENTANTES------"

#seleciono o representantes de cada componente gaussiano
for feature_index in xrange(len(featureBag)):
    if(fitProba[feature_index][fit[feature_index]] > max_proba_patchs[fit[feature_index]]):
        max_proba_patchs[fit[feature_index]] = fitProba[feature_index][fit[feature_index]]
        repre_patchs[fit[feature_index]] = getAppearance(localFeatureBagM[feature_index],win_x,win_y,img)


#-------------------------------------------------CONSTRUCAO-DO-PEP_REPRESENTATION-----------------

print "-----START BUILD PEP-Representation----------"
# print img.shape[0]
# print img.shape[1]
#realiza o somatorio de patchs 
for feature_index in xrange(len(featureBag)):
    componet = fit[feature_index]#seleciona a classe do exemplo
    componentRepre = repre_patchs[componet]#seleciona o componente que representa a classe
    
    local = getLocal(localFeatureBagM[feature_index])#pega o local de inicio do patch 
    partImg = componentRepre["a"]#pega a parte da imagem que representa a classe

    for i in xrange(int(local[0]),int(local[0]+win_x)):
        for j in xrange(int(local[1]),int(local[1]+win_y)):        
            pep_sum[i][j][0]+=partImg[i-int(local[0])][j-int(local[1])][0]#B
            pep_sum[i][j][1]+=partImg[i-int(local[0])][j-int(local[1])][1]#G
            pep_sum[i][j][2]+=partImg[i-int(local[0])][j-int(local[1])][2]#R
            countMatrix[i][j]+=1
        
#realiza a divisao dos pixels            
for i in xrange(img.shape[0]):
    for j in xrange(img.shape[1]):
        
        new_pixel = []
        new_pixel.append(pep_sum[i][j][0]/countMatrix[i][j])#B
        new_pixel.append(pep_sum[i][j][1]/countMatrix[i][j])#G
        new_pixel.append(pep_sum[i][j][2]/countMatrix[i][j])#R
        
        pep_r[i][j] = new_pixel
    
    
cv2.imshow("pep",pep_r)
cv2.imwrite("pep_repreGMMsave2.png", pep_r) 

cv2.waitKey()                
                            