import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import copy

jpy = pd.read_csv("JPN.csv", sep=",", encoding="shift-JIS")

train_period = (jpy["年"]<=2016)
test_period = (jpy["年"]>=2017)
interval = 25

def make_data(data):
    x = []
    y = []
    start = list(data["始値"])
    for i in range(len(start)-interval):
        ya = []
        ya.append(start[i+interval])
        xa = []
        for j in range(interval):
            d = i + j
                
            xa.append(start[d])
        x.append(xa)
        y.append(ya)
    x = np.array(x)
    y = np.array(y)
    return(x, y)


x_trains, y_trains = make_data(jpy[train_period])
x_tests, y_tests = make_data(jpy[test_period])

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout,  InputLayer
from keras.optimizers import Adam

model = Sequential()
model.add(InputLayer(input_shape=(25,)))
model.add(Dense(13, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='relu'))
model.compile(loss='mean_squared_error', optimizer=Adam(), metrics=['accuracy'])

model.summary()

history = model.fit(
    x_trains,
    y_trains,
    epochs=150,
    verbose=0,
)
print('ok')

pred = model.predict(x_tests)

import matplotlib.pyplot as plt

plt.plot(y_tests, label='real')
plt.plot(pred, label='pred')
plt.legend()
plt.savefig('long.jpg')
plt.show()
plt.plot(y_tests, label='real')
plt.plot(pred, label='pred')
plt.xlim([400,len(pred)])
plt.legend()
plt.savefig('short.jpg')
plt.show()
