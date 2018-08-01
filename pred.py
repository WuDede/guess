#coding: UTF-8
import numpy as np
import keras
import matplotlib.pyplot as plt
import pylab

TIME_STEPS = 10
BATCH_SIZE = 20
INPUT_SIZE = 1
OUTPUT_SIZE = 1
CELL_SIZE = 20
NR_POINT = 80
NR_PREDICT = 100
NR_TRAIN = 200

xd = np.linspace(0, NR_POINT - TIME_STEPS, num=NR_POINT -
                 TIME_STEPS, endpoint=False, dtype=int)

x_train = np.linspace(0, 8 * np.pi, num=NR_POINT)
x_train = np.sin(x_train).tolist() + 0.003 * xd


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        dataX.append(dataset[i:(i + look_back)])
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)


X, Y = create_dataset(x_train, TIME_STEPS)
X = np.reshape(X, (len(X), TIME_STEPS, INPUT_SIZE))

model = keras.Sequential()
model.add(keras.layers.LSTM(
    units=CELL_SIZE,
    input_shape=(TIME_STEPS, INPUT_SIZE)
))
model.add(keras.layers.Dense(OUTPUT_SIZE))

plt.rcParams['figure.figsize'] = (8.0, 4.0)
pylab.mpl.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

model.compile(optimizer='adam', loss='mse')
for i in range(NR_TRAIN):
    global p1, p2
    cost = model.train_on_batch(X, Y)
    pred = model.predict(X, 1)
    if i % 10 == 0 or i == NR_TRAIN:
        print("step: ", i, " cost: ", cost)
        plt.cla()
        plt.ylim(-3, 3)
        plt.xlim(0, NR_POINT + NR_PREDICT)
        p1, = plt.plot(xd, Y.flatten(), 'r')
        p2, = plt.plot(xd, pred.flatten(), 'b--')
        plt.legend([p1, p2], ['原始值', '训练值'])
        plt.text(1, 1.5, '训练轮次：%d\n训练损失：%f' % (i, cost))
        plt.draw()
        plt.pause(0.01)

xnd = []
ynd = []
for j in range(NR_PREDICT):
    x_pred = x_train[-TIME_STEPS:]
    x_pred = np.reshape(x_pred, (1, TIME_STEPS, INPUT_SIZE))
    p = model.predict(x_pred, 1)
    x_train.append(p[0, 0])
    xnd.append(NR_POINT - TIME_STEPS + j)
    ynd.append(p[0, 0])
    plt.ylim(-3, 3)
    plt.xlim(0, NR_POINT + NR_PREDICT)
    p3, = plt.plot(xnd, ynd, 'g')
    plt.legend([p1, p2, p3], ['原始值', '训练值', '预测值'])
    plt.draw()
    plt.pause(0.01)

plt.show()
