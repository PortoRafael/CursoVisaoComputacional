# -*- coding: utf-8 -*-
"""Visão Computacional: O Guia Completo - Redes Neurais para classificação de imagens.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12x-fvkVRo1SBUHgJ-XYSatNz2MXMYSzv

# Visão Computacional: O Guia Completo - Redes Neurais para classificação de imagens

# Abordagem 1: extração de todos os pixels da imagem

## Importação das bibliotecas
"""

import cv2
import numpy as np
import os
import zipfile
from google.colab.patches import cv2_imshow
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
tf.__version__

!pip install tensorflow == 2.6.0

"""## Extração dos pixels das imagens"""

from google.colab import drive
drive.mount('/content/drive')

path = '/content/drive/MyDrive/Cursos - recursos/Visão Computacional Guia Completo/Datasets/homer_bart_1.zip'
zip_object = zipfile.ZipFile(file = path, mode = 'r')
zip_object.extractall('./')
zip_object.close()

diretorio = '/content/homer_bart_1'
arquivos = [os.path.join(diretorio, f) for f in sorted(os.listdir(diretorio))]
print(arquivos)

type(arquivos)

largura, altura = 128, 128

128 * 128, 128 * 128 * 3

imagens = []
classes = []

imagem.shape

for imagem_caminho in arquivos:
  #print(imagem_caminho)
  try:
    imagem = cv2.imread(imagem_caminho)
    (H, W) = imagem.shape[:2]
  except:
    continue

  imagem = cv2.resize(imagem, (largura, altura))
  imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
  cv2_imshow(imagem)

  imagem = imagem.ravel()
  #print(imagem.shape)

  imagens.append(imagem)
  nome_imagem = os.path.basename(os.path.normpath(imagem_caminho))
  #print(nome_imagem)
  if nome_imagem.startswith('b'):
    classe = 0
  else:
    classe = 1

  classes.append(classe)
  print(classe)

imagens

print(classes)

imagens[100], classes[100]

imagens[200], classes[200]

type(imagens), type(classes)

X = np.asarray(imagens)
y = np.asarray(classes)

type(X), type(y)

X.shape

y.shape

sns.countplot(y);

np.unique(y, return_counts=True)

"""## Normalização dos dados"""

X[0].max(), X[0].min()

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X[0].max(), X[0].min()

X[1]

"""## Bases de treinamento e teste"""

X.shape

from sklearn.model_selection import train_test_split

X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size = 0.2, random_state = 1)

X_treinamento.shape, y_treinamento.shape

X_teste.shape, y_teste.shape

"""## Construção e treinamento da rede neural"""

128 * 128

(16384 + 2) / 2

# 16384 -> 8193 -> 8193
network1 = tf.keras.models.Sequential()
network1.add(tf.keras.layers.Dense(input_shape = (16384,), units = 8193, activation='relu'))
network1.add(tf.keras.layers.Dense(units = 8193, activation='relu'))
network1.add(tf.keras.layers.Dense(units = 1, activation='sigmoid'))

network1.summary()

# https://keras.io/api/optimizers/
# https://keras.io/api/losses/
network1.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

historico = network1.fit(X_treinamento, y_treinamento, epochs = 50)

"""## Avaliação da rede neural"""

historico.history.keys()

plt.plot(historico.history['loss']);

plt.plot(historico.history['accuracy']);

X_teste.shape

previsoes = network1.predict(X_teste)
previsoes

# 0 False Bart
# 1 True Homer
previsoes = (previsoes > 0.5)
previsoes

y_teste

from sklearn.metrics import accuracy_score
accuracy_score(y_teste, previsoes)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_teste, previsoes)
cm

sns.heatmap(cm, annot=True);

from sklearn.metrics import classification_report
print(classification_report(y_teste, previsoes))

"""## Salvar e carregar a rede neural"""

model_json = network1.to_json()
with open('network1.json', 'w') as json_file:
  json_file.write(model_json)

from keras.models import save_model
network1_saved = save_model(network1, 'weights1.hdf5')

with open('network1.json') as json_file:
  json_saved_model = json_file.read()
json_saved_model

network1_loaded = tf.keras.models.model_from_json(json_saved_model)
network1_loaded.load_weights('/content/weights1.hdf5')
network1_loaded.compile(loss = 'binary_crossentropy', optimizer = 'Adam', metrics = ['accuracy'])

network1_loaded.summary()

"""## Classificação de uma única imagem"""

X_teste[0], y_teste[0]

X_teste[0].shape

cv2_imshow(X_teste[0].reshape(128,128))

imagem_teste = X_teste[34]
imagem_teste = scaler.inverse_transform(imagem_teste.reshape(1, -1))

imagem_teste

cv2_imshow(imagem_teste.reshape(128,128))

network1_loaded.predict(imagem_teste)[0][0]

if network1_loaded.predict(imagem_teste)[0][0] < 0.5:
  print('Bart')
else:
  print('Homer')

"""# Abordagem 2: extração de características

## Extrator de características
"""

arquivos = [os.path.join(diretorio, f) for f in sorted(os.listdir(diretorio))]
print(arquivos)

export = 'boca,calca,sapatos,camisa,calcao,tenis,classe\n'

mostrar_imagens = False
caracteristicas = []

100 * 200

(2000 / 20000) * 100

for imagem_caminho in arquivos:
  #print(imagem_caminho)
  try:
    imagem_original = cv2.imread(imagem_caminho)
    (H, W) = imagem_original.shape[:2]
  except:
    continue

  imagem_alterada = imagem_original.copy()
  imagem_caracteristicas = []
  imagem_nome = os.path.basename(os.path.normpath(imagem_caminho))
  boca = calca = sapato = 0
  camisa = calcao = tenis = 0

  if imagem_nome.startswith('b'):
    classe = 0
  else:
    classe = 1

  for altura in range(0, H):
    for largura in range(0, W):
      # RGB -> BGR
      azul = imagem_alterada.item(altura, largura, 0)
      verde = imagem_alterada.item(altura, largura, 1)
      vermelho = imagem_alterada.item(altura, largura, 2)

      # Homer - marrom da boca
      if (azul >= 95 and azul <= 140 and verde >= 160 and verde <= 185 and vermelho >= 175 and vermelho <= 205):
        imagem_alterada[altura, largura] = [0, 255, 255]
        boca += 1

      # Homer - azul da calça
      if (azul >= 150 and azul <= 180 and verde >= 98 and verde <= 120 and vermelho >= 0 and vermelho <= 90):
        imagem_alterada[altura, largura] = [0, 255, 255]
        calca += 1

      # Homer - cinza dos sapatos
      if altura > (H / 2):
        if (azul >= 25 and azul <= 45 and verde >= 25 and verde <= 45 and vermelho >= 25 and vermelho <= 45):
          imagem_alterada[altura, largura] = [0, 255, 255]
          sapato += 1

      # Bart - laranja da camisa
      if (azul >= 11 and azul <= 50 and verde >= 85 and verde <= 105 and vermelho >= 240 and vermelho <= 255):
        imagem_alterada[altura, largura] = [0, 255, 128]
        camisa += 1

      # Bart - azul do calção
      if (azul >= 125 and azul <= 170 and verde >= 0 and verde <= 12 and vermelho >= 0 and vermelho <= 20):
        imagem_alterada[altura, largura] = [0, 255, 128]
        calcao += 1

      # Bart - azul do tênis
      if altura > (H / 2):
        if (azul >= 125 and azul <= 170 and verde >= 0 and verde <= 12 and vermelho >= 0 and vermelho <= 20):
          imagem_alterada[altura, largura] = [0, 255, 128]
          tenis += 1

  boca = round((boca / (H * W)) * 100, 9)
  calca = round((calca / (H * W)) * 100, 9)
  sapato = round((sapato / (H * W)) * 100, 9)
  camisa = round((camisa / (H * W)) * 100, 9)
  calcao = round((calcao / (H * W)) * 100, 9)
  tenis = round((tenis / (H * W)) * 100, 9)

  imagem_caracteristicas.append(boca)
  imagem_caracteristicas.append(calca)
  imagem_caracteristicas.append(sapato)
  imagem_caracteristicas.append(camisa)
  imagem_caracteristicas.append(calcao)
  imagem_caracteristicas.append(tenis)
  imagem_caracteristicas.append(classe)

  caracteristicas.append(imagem_caracteristicas)

  #print('Homer boca: ', imagem_caracteristicas[0], ' - Homer calça: ', imagem_caracteristicas[1], ' - Homer sapato: ', imagem_caracteristicas[2])
  #print('Bart camisa: ', imagem_caracteristicas[3], ' - Bart calção: ', imagem_caracteristicas[4], ' - Bart tênis: ', imagem_caracteristicas[5])
  
  f = (",".join([str(item) for item in imagem_caracteristicas]))
  export += f + '\n'

  if mostrar_imagens == True:
    imagem_alterada = cv2.cvtColor(imagem_alterada, cv2.COLOR_BGR2RGB)
    imagem_original = cv2.cvtColor(imagem_original, cv2.COLOR_BGR2RGB)
    figura, im = plt.subplots(1, 2)
    im[0].imshow(imagem_original)
    im[0].axis('off')
    im[1].imshow(imagem_alterada)
    im[1].axis('off')
    plt.show()


  #cv2_imshow(imagem_original)
  #print(H, W)
  #print(imagem_nome)

export

with open('features.csv', 'w') as file:
  for linha in export:
    file.write(linha)
file.closed

dataset = pd.read_csv('features.csv')
dataset

"""## Bases de treinamento e teste"""

X = dataset.iloc[:, 0:6].values
X

y = dataset.iloc[:, 6].values
y

from sklearn.model_selection import train_test_split
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size = 0.2, random_state = 1)

X_treinamento.shape, y_treinamento.shape

X_teste.shape, y_teste.shape

"""## Construção e treinamento da rede neural"""

(6 + 2) / 2

# 6 -> 4 -> 4 -> 4
network2 = tf.keras.Sequential()
network2.add(tf.keras.layers.Dense(input_shape = (6,), units = 4, activation='relu'))
network2.add(tf.keras.layers.Dense(units=4, activation='relu'))
network2.add(tf.keras.layers.Dense(units=4, activation='relu'))
network2.add(tf.keras.layers.Dense(units = 1, activation='sigmoid'))

network2.summary()

network2.compile(optimizer='Adam', loss='binary_crossentropy', metrics = ['accuracy'])

historico = network2.fit(X_treinamento, y_treinamento, epochs = 50)

"""## Avaliação da rede neural"""

historico.history.keys()

plt.plot(historico.history['loss']);

plt.plot(historico.history['accuracy']);

X_teste.shape

previsoes = network2(X_teste)
previsoes

previsoes = previsoes > 0.5
previsoes

y_teste

from sklearn.metrics import accuracy_score
accuracy_score(y_teste, previsoes)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_teste, previsoes)
cm

sns.heatmap(cm, annot=True);

from sklearn.metrics import classification_report
print(classification_report(y_teste, previsoes))

"""## Salvar, carregar e classificar uma única imagem"""

model_json = network2.to_json()
with open('network2.json','w') as json_file:
  json_file.write(model_json)

from keras.models import save_model
network2_saved = save_model(network2, '/content/weights2.hdf5')

with open('network2.json', 'r') as json_file:
  json_saved_model = json_file.read()
json_saved_model

network2_loaded = tf.keras.models.model_from_json(json_saved_model)
network2_loaded.load_weights('weights2.hdf5')
network2_loaded.compile(loss = 'binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

network2_loaded.summary()

imagem_teste = X_teste[0]
imagem_teste

imagem_teste.shape

imagem_teste = imagem_teste.reshape(1,-1)
imagem_teste.shape

network2_loaded.predict(imagem_teste)[0][0]

if network2_loaded.predict(imagem_teste)[0][0] < 0.5:
  print('Bart')
else:
  print('Homer')