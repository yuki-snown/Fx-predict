from sklearn.linear_model import LinearRegression
from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import copy

jpy = pd.read_csv("USDJPY.csv", sep=",", encoding="shift-JIS")

train_period = (jpy["年"]<=2016)
test_period = (jpy["年"]>=2017)
test_period1 = (jpy["年"]==2017)
test_period2 = (jpy["年"]==2018)
interval = 39
err = 0.5
day = 7

print(day, "日間のUSDJPY予想")

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
    print("最大誤差=", a, "平均誤差 = ", b, "ヒット率", c)
    
def check1(pred_y, test_x, day, err):
    i = 0
    j = 0
    for i in range(day):
        c = abs(pred_y[i-day] - test_x[i-day])
        if(c<=err):
            j = j+1
        i = i+1
    c = j/i
    print("hit = " ,c)

train_x1, train_y1 = make_data1(jpy[train_period])
test_x1, test_y1 = make_data1(jpy[test_period])
lr = LinearRegression(normalize=True)
lr.fit(train_x1, train_y1)
pred_y1 = lr.predict(test_x1)
pred_y1 = pred_y1.round(2)

train_x2, train_y2 = make_data1(jpy[train_period])
test_x2, test_y2 = make_data1(jpy[test_period1])
lr = LinearRegression(normalize=True)
lr.fit(train_x2, train_y2)
test_y2.append(np.nan)
for i in range(day):
    pred_y2 = lr.predict(test_x2)
    pred_y2 = pred_y2.round(2)
    del test_y2[-1]
    test_y2.append(pred_y2[-1])
    test_y2.append(np.nan)
    test_x22 = copy.copy(test_x2[-1])
    test_x22.append(pred_y2[-1])
    del test_x22[0]
    test_x2.append(test_x22)
    
check(pred_y2, test_y1, err)
check1(pred_y2, test_y1, day, err)
print("*赤字:実値 *青字:予想")

test_y3 = [0]
pred_y3 = [0]

for i in range(day):
    test_y3.append(test_y2[i-day])
    pred_y3.append(pred_y2[i-day])

del test_y3[0]
del pred_y3[0]


test_x4, test_y4 = make_data1(jpy[test_period2])
test_y5 = [0]
for i in range(day):
    test_y5.append(test_y4[i])
del test_y5[0]

fig,axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
axes[0,0].plot(test_y1, c='r')
axes[0,0].plot(pred_y1, c ='b')
axes[0,0].set_title("real")
axes[0,1].plot(test_y2, c='r')
axes[0,1].plot(pred_y2, c ='b')
axes[0,1].set_title("pred")
axes[1,0].plot(test_y1, c='r')
axes[1,0].plot(pred_y2, c ='b')
axes[1,0].set_title("comp")
axes[1,1].plot(pred_y3, c ='b')
axes[1,1].set_title("out")