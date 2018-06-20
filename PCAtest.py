# -*- coding: utf-8 -*-
import cv2
import openface
import os
import numpy as np
import glob

vector=[]
v=[]
align = openface.AlignDlib("../openface/models/dlib/shape_predictor_68_face_landmarks.dat")

#import image

pic = glob.glob('/home/docker/CK+image/*.png')
"""
pic=sorted(os.listdir("/home/docker/CK+image"))
pic.pop(0)
print(pic[:5])
"""
for p in pic:
    img=cv2.imread(p)
    bb = align.getLargestFaceBoundingBox(img)
    landmarks = align.findLandmarks(img, bb)

    aligned_face = align.align(64, img, bb, landmarks)
    aligned_face=cv2.cvtColor(aligned_face, cv2.COLOR_RGB2GRAY)
    aligned_face=cv2.equalizeHist(aligned_face)
    i=0

    #ベクトルを出す
    for a in aligned_face:
        for aa in a:
            v.append(aa)
        vector.append(v)
    
#PCA
from sklearn.decomposition import PCA
pca=PCA(n_components=10)
pca.fit(vector)
transformed=pca.fit_transform(features)
