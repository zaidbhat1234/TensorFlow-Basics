#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 18:17:14 2019

@author: zaidbhat
"""


import keras 
import tensorflow as tf

fashion_mnist = keras.datasets.fashion_mnist
(train_data,train_label),(test_data,test_label)=fashion_mnist.load_data()
train_data= train_data.reshape(60000,28,28,1)
train_data = train_data/255.0
test_data= test_data.reshape(10000,28,28,1)
test_data=test_data/255.0

model = keras.Sequential([
        keras.layers.Conv2D(64,(3,3),activation = 'relu', input_shape = (28,28,1)),
        keras.layers.MaxPooling2D(2,2),
        keras.layers.Conv2D(64,(3,3),activation='relu'),
        keras.layers.MaxPooling2D(2,2),
        keras.layers.Flatten(),
        keras.layers.Dense(128,activation='relu'),
        keras.layers.Dense(10,activation='softmax')])

model.compile(optimizer = tf.train.AdamOptimizer(),loss= 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit(train_data,train_label,epochs=10)
model.evaluate(test_data,test_label)