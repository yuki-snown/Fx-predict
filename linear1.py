from sklearn.linear_model import LinearRegression
from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

jpy = pd.read_csv("USDJPY.csv", sep=",", encoding="shift-JIS")

train_period = (jpy["年"]<=2016)
test_period = (jpy["年"]>=2017)
interval = 80
err = 0.5

def make_data1(data):
    x = []
    y = []
    start = list(data["始値"])
    for i in range(len(start)):
        if i <= interval: continue
        y.append(start[i])
        xa = []
        for j in range(interval):
            d = i + j - interval
            xa.append(start[d])
        x.append(xa)
    return(x, y)

def make_data2(data):
    x = []
    y = []
    start = list(data["高値"])
    for i in range(len(start)):
        if i <= interval: continue
        y.append(start[i])
        xa = []
        for j in range(interval):
            d = i + j - interval
            xa.append(start[d])
        x.append(xa)
    return(x, y)

def make_data3(data):
    x = []
    y = []
    start = list(data["安値"])
    for i in range(len(start)):
        if i <= interval: continue
        y.append(start[i])
        xa = []
        for j in range(interval):
            d = i + j - interval
            xa.append(start[d])
        x.append(xa)
    return(x, y)

def make_data4(data):
    x = []
    y = []
    start = list(data["終値"])
    for i in range(len(start)):
        if i <= interval: continue
        y.append(start[i])
        xa = []
        for j in range(interval):
            d = i + j - interval
            xa.append(start[d])
        x.append(xa)
    return(x, y)

def check(pred_y, test_y, err):
    i = 0
    j = 0
    k = []
    while i != len(pred_y):
        c = abs(pred_y[i]-test_y[i])
        if(c<=err):
            j = j + 1
        k.append(c)
        i= i + 1
    del(k[-1])
    a = max(k)
    b = sum(k) / len(k)
    c = j / i
    a = round(a, 2)
    b = round(b, 2)
    c = round(c, 2)
    print("平均誤差 = ", b)


train_x1, train_y1 = make_data1(jpy[train_period])
test_x1, test_y1 = make_data1(jpy[test_period])

lr = LinearRegression(normalize=True)
lr.fit(train_x1, train_y1)
pred_y1 = lr.predict(test_x1)
pred_y1 = pred_y1.round(2)
print("始値は", pred_y1[-1])
check(pred_y1, test_y1, err)

train_x2, train_y2 = make_data2(jpy[train_period])
test_x2, test_y2 = make_data2(jpy[test_period])

lr = LinearRegression(normalize=True)
lr.fit(train_x2, train_y2)
pred_y2 = lr.predict(test_x2)
pred_y2 = pred_y2.round(2)
print("高値は", pred_y2[-1])
check(pred_y2, test_y2, err)

train_x3, train_y3 = make_data3(jpy[train_period])
test_x3, test_y3 = make_data3(jpy[test_period])

lr = LinearRegression(normalize=True)
lr.fit(train_x3, train_y3)
pred_y3 = lr.predict(test_x3)
pred_y3 = pred_y3.round(2)
print("安値は", pred_y3[-1])
check(pred_y3, test_y3, err)

train_x4, train_y4 = make_data4(jpy[train_period])
test_x4, test_y4 = make_data4(jpy[test_period])

lr = LinearRegression(normalize=True)
lr.fit(train_x4, train_y4)
pred_y4 = lr.predict(test_x4)
pred_y4 = pred_y4.round(2)
print("終値は", pred_y4[-1])
check(pred_y4, test_y4, err)

print("*赤字:実値 *青字:予想")

fig,axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
axes[0,0].plot(test_y1, c='r')
axes[0,0].plot(pred_y1, c ='b')
axes[0,0].set_title("open")
axes[0,1].plot(test_y2, c='r')
axes[0,1].plot(pred_y2, c ='b')
axes[0,1].set_title("high")
axes[1,0].plot(test_y3, c='r')
axes[1,0].plot(pred_y3, c ='b')
axes[1,0].set_title("low")
axes[1,1].plot(test_y4, c='r')
axes[1,1].plot(pred_y4, c ='b')
axes[1,1].set_title("close")