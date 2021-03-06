# coding=utf-8
""" 
Created on Mon May 16 13:34:30 2016 
@author: Michelle 
"""
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.layers.advanced_activations import LeakyReLU
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np


def figures(history, figure_name="plots"):
    """ method to visualize accuracies and loss vs epoch for training as well as testind data\n 
        Argumets: history     = an instance returned by model.fit method\n 
                  figure_name = a string representing file name to plots. By default it is set to "plots" \n 
       Usage: hist = model.fit(X,y)\n              figures(hist) """
    from keras.callbacks import History
    if isinstance(history, History):
        hist = history.history
        epoch = history.epoch
        acc = hist['acc']
        loss = hist['loss']
        val_loss = hist['val_loss']
        val_acc = hist['val_acc']
        plt.figure(1)

        plt.subplot(221)
        plt.plot(epoch, acc)
        plt.title("Training accuracy vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")

        plt.subplot(222)
        plt.plot(epoch, loss)
        plt.title("Training loss vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        plt.subplot(223)
        plt.plot(epoch, val_acc)
        plt.title("Validation Acc vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Accuracy")

        plt.subplot(224)
        plt.plot(epoch, val_loss)
        plt.title("Validation loss vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Loss")
        plt.tight_layout()
        plt.show()
    else:
        print("Input Argument is not an instance of class History")


# part1: train data
# generate 100 numbers from -2pi to 2pi
x_train = np.linspace(-2 * np.pi, 2 * np.pi, 1000)  # array: [1000,]
x_train = np.array(x_train).reshape(
    (len(x_train), 1))  # reshape to matrix with [100,1]
# generate a matrix with size [len(x),1], value in (0,1),array: [1000,1]
n = 0.1 * np.random.rand(len(x_train), 1)
y_train = np.sin(x_train) + n

# 训练数据集：零均值单位方差
x_train = preprocessing.scale(x_train)
scaler = preprocessing.StandardScaler().fit(x_train)
y_train = scaler.transform(y_train)

# part2: test data
x_test = np.linspace(-5, 5, 2000)
x_test = np.array(x_test).reshape((len(x_test), 1))
y_test = np.sin(x_test)

# 零均值单位方差
x_test = scaler.transform(x_test)
#y_test = scaler.transform(y_test)
# plot testing data
#fig, ax = plt.subplots()
#ax.plot(x_test, y_test,'g')

# prediction data
x_prd = np.linspace(-3, 3, 101)
x_prd = np.array(x_prd).reshape((len(x_prd), 1))
x_prd = scaler.transform(x_prd)
y_prd = np.sin(x_prd)
# plot testing data
fig, ax = plt.subplots()
ax.plot(x_prd, y_prd, 'r')

# part3: create models, with 1hidden layers
model = Sequential()
model.add(Dense(100, init='uniform', input_dim=1))
# model.add(Activation(LeakyReLU(alpha=0.01)))
model.add(Activation('relu'))

model.add(Dense(50))
# model.add(Activation(LeakyReLU(alpha=0.1)))
model.add(Activation('relu'))

model.add(Dense(1))
# model.add(Activation(LeakyReLU(alpha=0.01)))
model.add(Activation('tanh'))

#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error',
              optimizer="rmsprop", metrics=["accuracy"])
#model.compile(loss='mean_squared_error', optimizer=sgd, metrics=["accuracy"])

#model.fit(x_train, y_train, nb_epoch=64, batch_size=20, verbose=0)
hist = model.fit(x_test, y_test, batch_size=10, nb_epoch=100,
                 shuffle=True, verbose=0, validation_split=0.2)
# print(hist.history)
score = model.evaluate(x_test, y_test, batch_size=10)

out = model.predict(x_prd, batch_size=1)
# plot prediction data

ax.plot(x_prd, out, 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
figures(hist)
