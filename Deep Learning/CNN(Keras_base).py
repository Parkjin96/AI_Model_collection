#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#참조 : https://lhh3520.tistory.com/376

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import utils, layers, datasets
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


#adjust data
batch_size = batch_size
num_classes = num_classes
epochs = epochs

train_images = train_images.astype('float32')
train_images = train_images / 255

test_images = test_images.astype('float32')
test_images = test_images / 255


# In[ ]:


#make model

model = keras.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(Dropout(0.25))


model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(Dropout(0.25))
model.add(layers.Dense(10, activation='softmax'))

model.summary()


# In[ ]:


#model compile
model.compile(
    loss = 'categorical_crossentropy',
    optimizer = 'adam',
    metrics = ['accuracy']
)


# In[ ]:


#train model
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10) # 오버피팅 방지

history = model.fit(
    train_imags, train_labels,
    epochos = epochs,
    validation_data = (val_imags, val_labels),
    shuffle = True,
    callbacks = [early_stopping]
)


# In[ ]:


# eval model
loss, acc = model.evaluate(val_imags, val_labels)

print('\nLoss: {}, ACC: {}'.format(loss,acc))


# In[ ]:


#predict
predictions = model.predict(test_images)

