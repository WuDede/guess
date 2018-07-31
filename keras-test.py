import tensorflow as tf
import keras
import numpy as np
import json
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt

# np.random.seed(2321908)


def get_color(x, y):
    # if x < 0.3 and y < 0.3:
    if((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5) < 0.04):

        return 1
    else:
        return 0


def loss_tf(y_true, y_pred):
    if y_true[0] == y_pred[0]:
        return tf.constant(0)
    else:
        return tf.constant(1)


data_x = np.random.rand(10000, 2)
data_y = []
for point in data_x:
    data_y.append(get_color(point[0], point[1]))
data_y = keras.utils.to_categorical(data_y, 2)

test_x = np.random.rand(10000, 2)
test_y = []
for point in test_x:
    c = get_color(point[0], point[1])
    test_y.append(c)
test_y = keras.utils.to_categorical(test_y, 2)

model = keras.models.Sequential()
model.add(Dense(64, input_shape=(2,), activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))

model.summary()

model.compile(optimizer=RMSprop(),
              loss='mse')  # , metrics=['accuracy'])
model.fit(x=data_x, y=data_y, batch_size=100, epochs=10)

pred_y = model.predict(test_x)
print(test_x)
print(test_y)
print(pred_y)
total = 0
testok = 0
colors = []
for t, p in zip(test_y, pred_y):
    total += 1
    if t[0] * p[0] + t[1] * p[1] > 0.5:
        testok += 1
        if t[0] == 0:
            colors.append('g')
        else:
            colors.append('b')
    else:
        colors.append('r')

print("%d / %d" % (testok, total))

keras.utils.plot_model(model)

#plt.scatter(test_x[:, 0], test_x[:, 1], c=colors, marker=',')
# plt.show()
