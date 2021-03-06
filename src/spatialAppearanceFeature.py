from os import listdir
import numpy as np
import cv2
from copy import copy
from cv2 import drawKeypoints
from cv2 import KeyPoint
from time import time
import pandas as pd
# from BDconnection import saveData
from scipy import spatial
from numba import jit
from _threading_local import local

def NormalizeAppearance(vector):
    normVector = []
    norma = np.linalg.norm(vector)
    for d in vector:
        normVector.append(d/norma)
    return normVector
 
def NormalizeLocal(coord,maxX,maxY):
    newCoord = []
    newCoord.append(coord[0]/maxX)
    newCoord.append(coord[1]/maxY)
    return newCoord

def crop_center(img,cropx,cropy):
    x = img.shape[0]
    y = img.shape[1]
    startx = (x/2)-(cropx/2)
    starty = (y/2)-(cropy/2)    
    return img[starty:starty+cropy,startx:startx+cropx]

def gaussianPyramid(img, num_layer):
    layers = []
    layers.append(img)
    for i_img in xrange(num_layer-1):
        img = cv2.pyrDown(layers[i_img])
        layers.append(img)
    return layers      

def OverlappingPatches(img,win_x,win_y):
    spatial_aparecen_features = []
    keypoints = []
    localFeatureNormalizado = []
    localFeature = []
    appearanceFeature =[]
    sift = cv2.SIFT()
    for i in xrange(0,img.shape[0],2):
        if((i+win_x) > img.shape[0]):
            break
        for j in xrange(0,img.shape[1],2):
            if((j+win_y)>img.shape[1]):
                break
            centerPatch = [i+((win_x-1.0)/2.0),j+((win_y-1.0)/2.0)]#[x,y]
            kp = KeyPoint(centerPatch[0],centerPatch[1],win_x)#(x,y,tam)
            keypoints.append(kp)
            localFeatureNormalizado.append(NormalizeLocal(centerPatch, float(img.shape[0]), float(img.shape[1])))
            localFeature.append(centerPatch)
                    
    local_index = 0
    
    kp,appearanceFeatureNoNorm = sift.compute(img,keypoints)#apearece feature
    
    for descritor in appearanceFeatureNoNorm:
        appearanceFeature.append(NormalizeAppearance(descritor))
    
    for index in xrange(len(localFeatureNormalizado)):
        al = appearanceFeature[index]+localFeatureNormalizado[index]
        spatial_aparecen_features.append(al)
    return spatial_aparecen_features,localFeature

def extract(path):
    
    print(path)
    img = cv2.imread(path)
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    
    imgCrop = crop_center(img, 150, 150)
    
    Gpy = gaussianPyramid(imgCrop, 3)

    sp_feature = []
    l_feature = []
    a = time()
    for gaus_layer in Gpy:
        feature,l = OverlappingPatches(gaus_layer, 8, 8)   
        sp_feature += feature
        l_feature += l
      
        
    b = time()
    print("time_extracao",(b-a))
    
    data = pd.DataFrame(sp_feature)
    data.columns = ["feature_%d" % x for x in xrange(130)]
    return data,l_feature



    
# imgNames = listdir("train")
# t = time()
# for name in imgNames:
#     data = extract(name)
#     t = time()
# #     saveData(data,name.strip('.jpg'))
#     t2 = time()
#     print('time save',(t2-t))
#     print("tam",len(data))
#     print('-'*50)
# t2 = time()
# print("tempoTotal",(t2-t))        
    
