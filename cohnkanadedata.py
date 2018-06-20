# -*- coding: utf-8 -*-
import csv
import cv2
import openface
import os

fileDir = os.path.dirname(os.path.realpath(__file__))
align = openface.AlignDlib("../openface/models/dlib/shape_predictor_68_face_landmarks.dat")
pic=sorted(os.listdir("/home/docker/cohnkanadepic"))
f=open('cohnkanade.csv','a')
writer = csv.writer(f)


for p in pic:
    lm=[]
    name="../cohnkanadepic/"+p
    print(p)
    img=cv2.imread(name)

    bb = align.getLargestFaceBoundingBox(img)
    landmarks= align.findLandmarks(img, bb)
    for l in landmarks:
        lm.append(l[0])
        lm.append(l[1])
    lm.insert(0,p)
    writer.writerow(lm)
