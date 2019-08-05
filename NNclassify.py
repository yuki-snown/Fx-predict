import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import copy
import os

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

#x_train
data_dir = 'x_trains/'
if not os.path.exists(data_dir):
        os.makedirs(data_dir)

comp = 0

for i in range(len(x_trains)):
    for j in range(interval):
        if i == 0 and j == 0:
            comp = x_trains[i][j]
            x_trains[i][j] = 1
        else :
            if comp > x_trains[i][j]: 
                comp = x_trains[i][j]
                x_trains[i][j] = 0
            elif comp <= x_trains[i][j] :
                comp = x_trains[i][j]
                x_trains[i][j] = 1
                
x_trains = np.reshape(x_trains, (len(x_trains), 5, 5))

#x_test
data_dir = 'x_tests/'
if not os.path.exists(data_dir):
        os.makedirs(data_dir)

comp = 0

for i in range(len(x_tests)):
    for j in range(interval):
        if i == 0 and j == 0:
            comp = x_tests[i][j]
            x_tests[i][j] = 1
        else :
            if comp > x_tests[i][j]: 
                comp = x_tests[i][j]
                x_tests[i][j] = 0
            elif comp <= x_tests[i][j] :
                comp = x_tests[i][j]
                x_tests[i][j] = 1
                
x_tests = np.reshape(x_tests, (len(x_tests), 5, 5))

#y_trains
comp = 0
for i in range(len(y_trains)):
    if i == 0:
        comp = y_trains[i][0]
        y_trains[i][0] = 0
    else :
        if comp > y_trains[i][0]: 
            comp = y_trains[i][0]
            y_trains[i][0] = 0
        elif comp <= y_trains[i][0] :
            comp = y_trains[i][0]
            y_trains[i][0] = 1

y_trains = list(map(int, y_trains))
y_trains = np.reshape(y_trains, (len(y_trains), 1))

#y_tests
comp = 0
for i in range(len(y_tests)):
    if i == 0:
        comp = y_tests[i][0]
        y_tests[i][0] = 0
    else :
        if comp > y_tests[i][0]: 
            comp = y_tests[i][0]
            y_tests[i][0] = 0
        elif comp <= y_tests[i][0] :
            comp = y_tests[i][0]
            y_tests[i][0] = 1

y_tests = list(map(int, y_tests))
y_tests = np.reshape(y_tests, (len(y_tests), 1))

from keras.models import Sequential
from keras.layers import InputLayer, Dense
from keras.layers.recurrent import LSTM
from keras import optimizers, regularizers

model = Sequential()

model.add(InputLayer(input_shape=(5, 5)))
weight_decay = 1e-4
model.add(LSTM(units=128, dropout=0.25, return_sequences=True))
model.add(LSTM(units=128, dropout=0.25, return_sequences=True))
model.add(LSTM(units=128, dropout=0.25, return_sequences=False, kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Dense(units=1, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(), metrics=['accuracy'])

model.summary()

history = model.fit(
    x_trains,
    y_trains,
    epochs=10,
    batch_size=100,
    verbose=1,
    validation_data=(x_tests, y_tests)
)

# 汎化制度の評価・表示
score = model.evaluate(x_tests, y_tests, batch_size=32, verbose=0)
print('validation loss:{0[0]}\nvalidation accuracy:{0[1]}'.format(score))