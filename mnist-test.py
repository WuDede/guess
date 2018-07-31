#!/usr/bin/python
# coding=utf-8
'''
Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''


from __future__ import print_function  # 使python2.7可以使用print函数形式

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop


batch_size = 128
num_classes = 10
epochs = 20

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255  # 输入的 x 变成 60,000*784 的数据，然后除以 255 进行标准化，因为每个像素都是在 0 到 255 之间的，标准化之后就变成了 0 到 1 之间
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
# 把 y 变成了 one-hot 的形式，即之前 y 是一个数值， 在 0-9 之间，现在是一个大小为 10
# 的向量，它属于哪个数字，就在哪个位置为 1，其他位置都是 0
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# build the neural net
model = Sequential()  # 序贯式模型
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))  # 防止过拟合，每次更新参数时随机断开一定百分比（rate）的输入神经元
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.summary()  # 打印出模型概况


# compile the model
model.compile(loss='categorical_crossentropy',  # 对数损失
              optimizer=RMSprop(),
              metrics=['accuracy'])


# train the model
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,  # 日志显示
                    validation_data=(x_test, y_test))  # fit将模型训练epochs轮


# test the model
# evaluate函数按batch计算在某些输入数据上模型的误差
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
