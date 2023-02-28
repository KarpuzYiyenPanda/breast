import numpy as np
import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, Add, BatchNormalization, Lambda, Concatenate, Reshape
from keras.models import Model
import tensorflow as tf
import os
import pandas as pd
import pydicom as dicom
import matplotlib.pyplot as plt
import math
from keras_preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import pydicom
import cv2
import os

def resize_dcm(dcmdosyası, boyut):
    dcmgörsel = pydicom.read_file(dcmdosyası)
    img = dcmgörsel.pixel_array
    img = cv2.resize(img, boyut, interpolation = cv2.INTER_AREA)
    dcmgörsel.PixelData = img.tobytes()
    dcmgörsel.Rows, dcmgörsel.Columns = img.shape
    return dcmgörsel

folder = "C:/Users/USER/Desktop/data"

ana1 = [f.path for f in os.scandir(folder) if f.is_dir()]

for ana2 in ana1:
    görseller = [f.path for f in os.scandir(ana2) if f.is_file()]

    for görsellercücük in görseller:
        resize2 = resize_dcm(görsellercücük, (800, 800))
        yeni = os.path.join(ana2, os.path.basename(görsellercücük))
        pydicom.write_file(yeni, resize2)



inputs = Input(shape=(800,800,1))
input2 = Input(shape=(1))
input3 = Input(shape=(1))

x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
x = BatchNormalization()(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)

x = MaxPooling2D((2, 2), strides=(2, 2))(x)
x = Dropout(0.5)(x)

x = Conv2D(128, (3, 3), activation='relu', strides=2, padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3, 3), activation='relu', strides=2, padding='same')(x)
x = BatchNormalization()(x)

res1 = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
res2 = Conv2D(128, (3, 3), activation='relu', padding='same')(res1)
x = Add()([x, res2])

x = MaxPooling2D((2, 2), strides=(2, 2))(x)
x = Dropout(0.5)(x)

x = Conv2D(256, (3, 3), activation='relu', strides=2, padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(256, (3, 3), activation='relu', strides=2, padding='same')(x)
x = BatchNormalization()(x)

res1 = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
res2 = Conv2D(256, (3, 3), activation='relu', padding='same')(res1)
x = Add()([x, res2])

res1 = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
res2 = Conv2D(256, (3, 3), activation='relu', padding='same')(res1)
x = Add()([x, res2])

x = MaxPooling2D((2, 2), strides=(2, 2))(x)
x = Conv2D(512, (3, 3), activation='relu', strides=2, padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(512, (3, 3), activation='relu', strides=2, padding='same')(x)
x = BatchNormalization()(x)

res1 = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
res2 = Conv2D(512, (3, 3), activation='relu', padding='same')(res1)
x = Add()([x, res2])

res1 = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
res2 = Conv2D(512, (3, 3), activation='relu', padding='same')(res1)
x = Add()([x, res2])

x = Flatten()(x)
x = Concatenate()([x,input2,input3])
x = Dense(256, activation='relu')(x)
x = Dense(256, activation='relu')(x)
output = Dense(4, activation='softnax')(x) #birads score
output2 = Dense(4, activation='softmax')(x) #kompozisyon
output3 = Dense(2, activation='softmax')(x) #bolge 1
output4 = Dense(2, activation='softmax')(x) #bolge 2
output5 = Dense(2, activation='softmax')(x) #bolge 3


model = Model([inputs, input2,input3], [output,output2, output3, output4, output5])
model.compile(optimizer='adam',
              loss={'dense_2': 'binary_crossentropy',
                    'dense_3': 'binary_crossentropy',
                    'dense_4': 'binary_crossentropy',
                    'dense_5': 'binary_crossentropy',
                    'dense_6': 'binary_crossentropy'},
              metrics=['accuracy'])

model.summary()



data = pd.read_excel('C:/Users/USER/Desktop/asdsadasdas.xlsx')
data.rename(columns={'HASTANO': 'ID', 'BIRADS KATEGORİSİ':'birads', 'MEME KOMPOZİSYONU': 'comp', 'KADRAN BİLGİSİ (SAĞ)': 'areaR', 'KADRAN BİLGİSİ (SOL)': 'areaL'}, inplace=True)
data.replace(['BI-RADS0', 'BI-RADS1-2', 'BI-RADS4-5'], [0,1,2], inplace=True)
data.drop(data.columns[5], inplace=True, axis=1)
data['comp'].value_counts()
data.fillna(0, inplace=True)
data.replace(['A', 'B', 'C', 'D'], [0,1,2,3], inplace=True)



def transform_to_hu(medical_image, image):

    intercept = medical_image.RescaleIntercept

    slope = medical_image.RescaleSlope

    hu_image = image * slope + intercept

    return hu_image

def pad_images(images, target_shape):
    padded_images = []
    for image in images:
        padded_image = np.pad(image, [(0, target_shape[0] - image.shape[0]), (0, target_shape[1] - image.shape[1])], mode='constant')
        padded_images.append(padded_image)
    return np.array(padded_images)


def train():
    mainDir = 'C:/Users/USER/Desktop/data/'
    for d in data.iterrows():
        patientDir = mainDir + str(d[1][0])
        for f in os.listdir(patientDir):
            yon = 1 if f[0] == 'R' else 0
            view = 1 if f[1] == 'M' else 0
            dcm = dicom.dcmread(patientDir + '/' + f)
            image = transform_to_hu(dcm, dcm.pixel_array)
            o1 = 0
            o2 = 0
            o3 = 0
            areaStr = d[1]['area' + ('R' if yon == 1 else 'L')]
            if areaStr!= 0:
                areaStr = areaStr.replace('[', '').replace(']', '')
                lst = areaStr.split(',')
                newLst = list(map(lambda x : x.replace('"', ''), lst))
                print(newLst)
                for a in newLst:
                    if(a == 'MERKEZ'):
                        o2 = 1
                        continue
                    t = a.split(' ')
                    if(view == 1): #mlo
                        if t[0] == 'ÜST':
                            o1 = 1
                        else:
                            o3 = 1
                    else:
                        if t[1] == 'DIŞ':
                            o1 = 1
                        else:
                            o3 = 1
            print(o1, o2, o3, sep=', ')

            birads_panda = to_categorical(d[1]['birads'], num_classes=4)
            comp_panda = to_categorical(d[1]['comp'], num_classes=4)
            o1_panda = to_categorical(o1, num_classes=2)
            o2_panda = to_categorical(o2, num_classes=2)
            o3_panda = to_categorical(o3, num_classes=2)

            y = (np.array([birads_panda]), np.array([comp_panda]), np.array([o1_panda]), np.array([o2_panda]), np.array([o3_panda]))

            x = (np.array([image]), np.array([yon]), np.array([view]))

            model.fit(x,y,epochs=5)

train()

























