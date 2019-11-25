#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 17:54:57 2019

@author: zaidbhat
"""


import import keras
import tensorflow as tf
import matplotlib.pyplot as plt


class myCallBack(keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs=[]):
        if(logs.get('loss')<0.4):
            print('We reached result early')
            self.model.stop_training = True

callbacks = myCallBack()

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images,test_labels) = fashion_mnist.load_data()

model = keras.Sequential([
        keras.layers.Flatten(input_shape= (28,28)),
        keras.layers.Dense(128, activation = tf.nn.relu),
        keras.layers.Dense(10, activation = tf.nn.softmax)])

plt.imshow(train_images[42])


train_images = train_images / 255.0
test_images = test_images / 255.0
print(train_images[42])

model.compile(optimizer = tf.train.AdamOptimizer(),loss= 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit(train_images,train_labels,epochs = 100,callbacks=[callbacks])

model.evaluate(test_images,test_labels)