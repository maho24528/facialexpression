# -*- coding: utf-8 -*-
import cv2
import openface
import os

fileDir = os.path.dirname(os.path.realpath(__file__))
align = openface.AlignDlib("../openface/models/dlib/shape_predictor_68_face_landmarks.dat")

#import image
img=cv2.imread("img_000000.png")
imgname="img_000000"

bb = align.getLargestFaceBoundingBox(img)
landmarks = align.findLandmarks(img, bb)
aligned_face = align.align(255, img, bb, landmarks)

#画像の縮小
aligned_face=cv2.resize(aligned_face,(200,200))
print(aligned_face[0])

#左右反転
flipimg = cv2.flip(aligned_face, 1)

#右上に正規化した画像、左右反転した画像を貼り付け
img[0:200,0:200]=aligned_face
img[0:200,201:401]=flipimg

#顔の特徴点に円をかく
for a in landmarks:
    cv2.circle(img,a,5,(225,0,0),-1)

#画像書き込み
cv2.imwrite("twoimages.jpg",img)
