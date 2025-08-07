# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 17:55:48 2025

@author: ENVER
"""

import pandas as pd
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
import os
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


train_dir = 'data/train/'
file_list = os.listdir(train_dir)
filepaths = []
labels = []

for name in file_list:
    full_dir = train_dir + name
    filepaths.append(full_dir)
    if name.startswith('cat'):
        labels.append(0)
    elif name.startswith('dog'):
        labels.append(1)
        
        
data_dict= {
    'filepath': filepaths,
    'label': labels,

    
    }
        
        
df = pd.DataFrame(data_dict)
df = df.sample(frac=1).reset_index(drop=True)
early_stopping = EarlyStopping(monitor='val_loss',      
    patience=5,             
    restore_best_weights=True)


train_df,val_df = train_test_split(df, train_size=0.8,random_state=(42),stratify=df['label'])

img_size = 150
batch_size = 32

def parse_image(filepath,label):
    image_string = tf.io.read_file(filepath)
    image = tf.image.decode_jpeg(image_string,channels=3)
    image = tf.image.resize(image,[img_size,img_size])
    image = image / 255.0
    return image,label


train_dataset = tf.data.Dataset.from_tensor_slices((train_df['filepath'],train_df['label']))
train_dataset = train_dataset.map(parse_image,num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)


val_dataset = tf.data.Dataset.from_tensor_slices((val_df['filepath'],val_df['label']))
val_dataset = val_dataset.map(parse_image,num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

data_manipulation = keras.Sequential([
    
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
    
    ])

model = keras.Sequential([
    keras.Input(shape=(150,150,3)),
    data_manipulation,
    layers.Conv2D(32,kernel_size=3,activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64,kernel_size=3,activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128,kernel_size=3,activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(256,kernel_size=3,activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(), 
    layers.Dense(256,activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1,activation='sigmoid')
    
    
    ])

model.summary()

model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy']
    

    )

epochs = 25

history = model.fit(train_dataset,epochs = epochs,validation_data = val_dataset, callbacks = [early_stopping] )


score = model.evaluate(val_dataset,verbose=0)
print(score[1])
model.save("cat_dogs.keras")

