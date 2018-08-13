# coding:utf-8
import numpy as np
# import matplotlib.pyplot as plt
import keras
import pandas

np.random.seed(20131118)


def splite_base(data, base=10):
    # 将数据转换为不同的进制，然后每个位单独一个元素，每个数据一个list
    # 返回多个元素构成的DataFrame
    def get_nr_ipi(x, b):
        i = 1
        while x / b >= 1:
            x /= b
            i += 1
        return i
    nr_ipi = get_nr_ipi(data.max(), base)
    ret = []
    for d in data:
        t = np.zeros(nr_ipi, dtype='int')
        for i in range(nr_ipi):
            a = d % base
            t[nr_ipi - 1 - i] = a
            d = int(d / base)
        ret.append(t.tolist())
    return pandas.DataFrame(ret)


def read_ori_data(path):
    data = pandas.read_csv(path)
    return data


def conv_data_to_base(data, base=10, lookback=1):
    print(data)
    # 将数据转换为监督训练数据
    conved = pandas.DataFrame(dtype='int')
    rs, re, be = 0, 0, 0
    for col in data.columns:
        conved = pandas.concat([conved, splite_base(data[col], base)], axis=1)
        if col == 'yno':
            # 红号开始的列
            rs = conved.shape[1]
        elif col == 'r6':
            # 红号结束的列
            re = conved.shape[1]
        elif col == 'b':
            # 蓝号结束的列
            be = conved.shape[1]
    x, yr, yb = [], [], []
    for i in range(len(conved) - lookback):
        tx = conved.iloc[i + lookback, :].tolist()[0:rs]
        tr = conved.iloc[i + lookback, :].tolist()[rs:re]
        tb = conved.iloc[i + lookback, :].tolist()[re:be]
        for j in range(lookback):
            tx += conved.iloc[i + j, :].tolist()[rs:be]
        x.append(tx)
        yr.append(tr)
        yb.append(tb)
    return np.array(x), np.array(yr), np.array(yb)


ax, ar, ab = conv_data_to_base(read_ori_data("ssq.csv").head(10))
print(ax.shape)
print(ax)
print(ar.shape)
print(ar)
print(ab.shape)
print(ab)
exit(0)
# ax = keras.utils.np_utils.to_categorical(ax)
# ar = keras.utils.np_utils.to_categorical(ar)
# ab = keras.utils.np_utils.to_categorical(ab)
# print(ax.shape)
# print(ar.shape)
# print(ab.shape)

model = keras.models.Sequential()

model.add(keras.layers.Dense(
    units=64,
))
model.add(keras.layers.Dropout(
    rate=0.2
))

model.add(keras.layers.Dense(
    units=64,
))
model.add(keras.layers.Dropout(
    rate=0.2
))


model.add(keras.layers.Dense(
    units=14,
))
