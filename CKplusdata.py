# -*- coding: utf-8 -*-
import csv
import cv2
import openface
import os
import numpy
import glob 

fileDir = os.path.dirname(os.path.realpath(__file__))
align = openface.AlignDlib("../openface/models/dlib/shape_predictor_68_face_landmarks.dat")
pic=sorted(os.listdir("/home/docker/CK+image"))
f=open('CKplus.csv','a')
writer = csv.writer(f)


print(pic[0])
pic.pop(0)

#奇数行は真顔
pic_first=pic[0::2]
#偶数行は変化後の顔
pic_last=pic[1::2]

print(pic_first)
print(pic_last)


#変化の前後の特徴点を比較
#landmarksの0と16が顔の横幅
for i in range(len(pic_first)):
    lm=[]

    name1="../CK+image/"+pic_first[i]
    name2="../CK+image/"+pic_last[i]

    img1=cv2.imread(name1)
    img2=cv2.imread(name2)

    bb1 = align.getLargestFaceBoundingBox(img1)
    bb2 = align.getLargestFaceBoundingBox(img2)

#比較する２つのlandmarks
    lm1= align.findLandmarks(img1, bb1)
    lm2= align.findLandmarks(img2, bb2)

#横幅で割る
    a=numpy.array(lm1[16])
    b=numpy.array(lm1[0])
    facesize=numpy.linalg.norm(a-b)

    for r in range(len(lm2)):
        lm.append((lm2[r][0]-lm1[r][0])/facesize)
        lm.append((lm2[r][1]-lm1[r][1])/facesize)

    lm.insert(0,pic_first[i])
    writer.writerow(lm)
