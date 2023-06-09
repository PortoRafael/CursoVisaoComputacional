# -*- coding: utf-8 -*-
"""Olá, este é o Colaboratory

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/notebooks/intro.ipynb
"""

import cv2
from google.colab.patches import cv2_imshow

img = cv2.imread('/content/drive/MyDrive/Machine Learning/Visão Computacional Guia Completo/My_Images/Pessoas/R.jpg')
img = cv2.resize(img,
                 (1020,800))

#cv2_imshow(img)
img2 = img.copy()
img_cinza = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#cv2_imshow(img_cinza)

detector_facial = cv2.CascadeClassifier('/content/drive/MyDrive/Machine Learning/Visão Computacional Guia Completo/Cascades/haarcascade_frontalface_default.xml')
detector_olhos = cv2.CascadeClassifier('/content/drive/MyDrive/Machine Learning/Visão Computacional Guia Completo/Cascades/haarcascade_eye.xml')



dtc_Face = detector_facial.detectMultiScale(img_cinza,
                                       scaleFactor = 1.005,
                                       minNeighbors = 7,
                                       minSize = (123,123),
                                       maxSize = (200,200)
                                       )

dtc_Olhos = detector_olhos.detectMultiScale(img_cinza,
                                       scaleFactor = 1.03,
                                       minNeighbors = 20,
                                       minSize = (26,26),
                                       maxSize = (51,51)
                                       )

for x,y,w,h in dtc_Face:
  print(f'Face: {w} - {h}')
  if (w,h) != (125,125):
    cv2.rectangle(img2,(x,y),(x+w,y+h), (0,255,0),4)

for x,y,w,h in dtc_Olhos:
  print(f'Olhos: {w} - {h}')
  cv2.rectangle(img2,(x,y),(x+w,y+h),(0,0,255),1)

cv2_imshow(img2)