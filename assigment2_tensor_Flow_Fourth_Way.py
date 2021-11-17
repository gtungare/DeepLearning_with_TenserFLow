"""
Name - Gaurav Tungare
Class: CS 767 - Fall 1
Date: Nov 2021
Homework  # Assignment 2 - Fourth Way

"""

import tensorflow as tf
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from tensorflow.keras.utils import  to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten,Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import  classification_report,confusion_matrix
from tensorflow.keras.metrics import Precision,AUC,Recall

import seaborn as sns

mnist = tf.keras.datasets.mnist # one of a handful of data sets known to Keras/TensorFlow

print(type(mnist))

# mnist.load_data() produces a pair of inpu/output tensors for training
# and one for testing.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)

single_image=x_train[0]
print(single_image)

plt.imshow(single_image)
#plt.show()
print(y_train)
print(y_train.shape)
y_example=to_categorical(y_train)
print(y_example.shape)

print(y_example[0])

y_cat_test=to_categorical(y_test)
y_cat_train=to_categorical(y_train)
print(single_image.min())
print(single_image.max())
x_train=x_train/255
x_test=x_test/255
scaled_image=x_train[0]
#print(scaled_image)
plt.imshow(scaled_image)
plt.show()

print(x_train.shape)
# batch_size,width,height,color_Channels
x_train=x_train.reshape(60000, 28, 28,1)
x_test=x_test.reshape(10000, 28, 28,1)

model= Sequential()
## Added
model.add(Conv2D(filters=16,kernel_size=(4,4),strides=(1,1),padding='valid', input_shape=(28,28,1),activation='relu'))
## Added
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters=8,kernel_size=(4,4),strides=(1,1),padding='valid',activation='relu'))
## Added
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.2))
## output layer since this is a multi class problem
model.add(Dense(10,activation='softmax'))
sgd = tf.keras.optimizers.SGD(learning_rate=0.02)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
early_stop= EarlyStopping(monitor='val_loss',patience=1)
model.fit(x_train,y_cat_train,epochs=15,validation_data=(x_test,y_cat_test),callbacks=[early_stop])



metrics = pd.DataFrame(model.history.history)
print(metrics)
metrics[['loss','val_loss']].plot()
plt.show()

metrics[['accuracy','val_accuracy']].plot()
plt.show()

model.evaluate(x_test,y_cat_test,verbose=0)
prediction= np.argmax(model.predict(x_test), axis=-1)

print(classification_report(y_test,prediction))
confusion_matrix(y_test,prediction)
plt.figure(figsize=(10,6))
sns.heatmap(confusion_matrix(y_test,prediction),annot=True)

my_number=x_test[0]
plt.imshow(my_number.reshape(28,28))
plt.show()

## batch siz, width,height,color channe

y_predict = np.argmax(model.predict(my_number.reshape(1,28,28,1)), axis=-1)
print(y_predict)

