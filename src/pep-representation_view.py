import cv2
import pandas as pd
from time import time
from os import listdir
from sklearn.mixture import GMM
from spatialAppearanceFeature import crop_center
from spatialAppearanceFeature import extract
from copy import copy

def getAppearance(coord,win_x,win_y,img):
    linha = coord[1] - ((win_x-1)/2)
    coluna = coord[0] - ((win_y-1)/2)
   
    
    a = img[linha:linha+win_x,coluna:coluna+win_y]
#     cv2.imshow('sa',a)
#     cv2.waitKey()        
    return{"a":a,"l":[linha,coluna]}
    
    
# treinamento
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

print "-------START TRAIN-------"
  
num_components = 15
GMM = GMM(n_components=num_components,covariance_type='spherical', n_iter=20)
t1 = time() 
GMM.fit(data)
t2 = time()
print "time fit",(t2-t1)

print "-------START CLASSIFICATION-------"

featureBag = extract("George_W_Bush_0179.jpg")# features
featureBagM = featureBag.as_matrix()     
localFeatureBagM = extract("George_W_Bush_0179.jpg",True).as_matrix()   
fitProba = GMM.predict_proba(featureBag) #probabilidades de cada exemplo por cada componet
fit = GMM.predict(featureBag)#classificacao do cojunto de treino  

#parametros
win_x = 8
win_y = 8        
K = 15

#estruturas auxiliares
print "--------START AUX STRUCT---------"

img = cv2.imread("George_W_Bush_0179.jpg")
img = crop_center(img,150,150)
pep_r = copy(img)
countMatrix = [[ 0 for _ in xrange(150)] for _ in xrange(150)]
for i in xrange(150):
    for j in xrange(150):
        pep_r[i][j][0] = 0
        pep_r[i][j][1] = 0
        pep_r[i][j][2] = 0
        
repre_patchs = [{"a":None,"l":[0,0]} for _ in xrange(K)]#guarda o pedaco da img e o local do patch que representa o componente k
max_proba_patchs = [0 for _ in xrange(K)]#guarda as maiores probas para cada k component

print "-------SELECT REPRESENTANTES------"

#seleciono o representantes de cada componente gaussiano
for feature_index in xrange(len(featureBag)):
    if(fitProba[feature_index][fit[feature_index]] > max_proba_patchs[fit[feature_index]]):
        max_proba_patchs[fit[feature_index]] = fitProba[feature_index][fit[feature_index]]
        repre_patchs[fit[feature_index]] = getAppearance(localFeatureBagM[feature_index][-2:],win_x,win_y,img)

print "-----START BUILD PEP-Representation----------"

#realiza o somatorio de patchs 
for feature_index in xrange(len(featureBag)):
    componet = fit[feature_index]#seleciona a classe do exemplo
    componentRepre = repre_patchs[componet]#seleciona o componente que representa a classe
    local = componentRepre["l"]
    partImg = componentRepre["a"]
    for i,x in zip(xrange(int(local[0]),int(local[0]+win_x)),xrange(8)):
        for j,y in zip(xrange(int(local[1]),int(local[1]+win_y)),xrange(8)):
            print "Y:",y,"-"*50
            print pep_r[i][j][0],"+",partImg[x][y][0]
            print pep_r[i][j][1],"+",partImg[x][y][1]
            print pep_r[i][j][2],"+",partImg[x][y][2]
            pep_r[i][j][0]+=partImg[x][y][0]
            pep_r[i][j][1]+=partImg[x][y][1]
            pep_r[i][j][2]+=partImg[x][y][2]

            countMatrix[i][j]+=1
        a = raw_input(">>")
          
#realiza a divisao dos pixels            
for i in xrange(150):
    for j in xrange(150):
        pep_r[i][j][0] = pep_r[i][j][0]/countMatrix[i][j]
        pep_r[i][j][1] = pep_r[i][j][1]/countMatrix[i][j]
        pep_r[i][j][2] = pep_r[i][j][2]/countMatrix[i][j]

cv2.imshow("pep",pep_r)    
cv2.waitKey()                
                            