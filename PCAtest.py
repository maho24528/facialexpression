# -*- coding: utf-8 -*-
import csv
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

    #ベクトルを出す
    for a in aligned_face:
        for aa in a:
            v.append(aa)
    vector.append(v)
    v=[]

#PCA
from sklearn.decomposition import PCA
pca=PCA(n_components=10)
pca.fit(vector)
X=pca.transform(vector)

#転置
#X=X.T

#書き込み
xxx=[]
f=open('PCAresult2.csv','a')
writer = csv.writer(f)
for x in X:
    for xx in x:
        xxx.append(xx)
    writer.writerow(xxx)
    xxx=[]
