#coding: UTF-8
import numpy as np
import keras
import pandas
from sklearn.preprocessing import OneHotEncoder

# 年,月,日,期号,红1,红2,红3,红4,红5,红6,蓝,销售额,一等奖,二等奖


def read_data():
    data = pandas.read_csv('ssq.csv')
    return data


def get_dataset():
    dataset = read_data()
    return dataset['蓝']


def gen_train_data(data, mem_term):
    # input: like data=[1 2 3 4 5 6], mem_term = 3
    # output: like
    # 1 2 3 -> 4
    #   2 3 4 -> 5
    #     3 4 5 -> 6
    x, y = [], []
    for i in range(len(data) - mem_term):
        x.append(data[i:i + mem_term].values.tolist())
        y.append(data[i + mem_term])
    return np.array(x), y


TRAIN_SIZE = 1000
BATCH_SIZE = 8
LAYER_CELLS_LIST = [8, 32, 32, 16]
LAYER_ACTIV_LIST = ['relu', 'relu']
TIME_STEP = 10
LR = 0.001

blueall = get_dataset()
train_x = blueall[0:TRAIN_SIZE]
test_x = blueall[TRAIN_SIZE:]

Train_X, Train_Y_pre = gen_train_data(train_x, TIME_STEP)
Test_X, Test_Y = gen_train_data(test_x, TIME_STEP)
Train_X = np.reshape(Train_X, (len(Train_X), TIME_STEP, 1))
Train_Y = np.zeros((len(Train_Y_pre), 16), dtype=int)
for yi in range(len(Train_Y_pre)):
    Train_Y[yi][Train_Y_pre[yi] - 1] = 1

print(Train_Y.head)
print(Train_Y_pre.head)
exit(0)
Test_X = np.reshape(Test_X, (len(Test_X), TIME_STEP, 1))

model = keras.Sequential()
# model.add(keras.layers.Dense(
#     units=LAYER_CELLS_LIST[0],
#     activation=LAYER_ACTIV_LIST[0]
# ))
model.add(keras.layers.LSTM(
    units=LAYER_CELLS_LIST[1],
    input_shape=(TIME_STEP, 1)
))
model.add(keras.layers.LSTM(
    units=LAYER_CELLS_LIST[2]
))
model.add(keras.layers.Dense(
    units=LAYER_CELLS_LIST[2],
    activation='softmax'
))
