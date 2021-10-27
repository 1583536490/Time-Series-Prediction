import keras
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPool1D, Flatten
import numpy as np
import pandas as pd
from pylab import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
# import os
# os.environ["TF_KERAS"] = '1'

#读取数据集
df = pd.read_csv('./purchase_seq_201402_201407.csv', parse_dates=['report_date'], index_col='report_date')
data = df['value'].values

#数据集切分、数据集归一化
train_size = 120
train_set = data[:train_size].reshape(-1,1)
scaler = StandardScaler().fit(train_set)
norm_train_data = StandardScaler().fit_transform(train_set)
x_train, y_train = [], []
step_size = 21
for i in range(step_size, train_size):
    x_train.append(norm_train_data[i - step_size:i, 0])
    y_train.append(norm_train_data[i, 0])

x_test, y_test = [], []
test_data = data[len(train_set)-step_size:].reshape(-1,1)
norm_test_data = scaler.transform(test_data)
for i in range(step_size, len(data)-train_size+step_size):
    x_test.append(norm_test_data[i-step_size:i, 0])
    y_test.append(norm_test_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train, y_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1), \
                   y_train.reshape(y_train.shape[0], 1, 1)
x_test, y_test = np.array(x_test), np.array(y_test)
x_test, y_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1), \
                   y_test.reshape(y_test.shape[0], 1, 1)

#模型搭建
model = Sequential()
model.add(Conv1D(10, kernel_size=3, strides=2, activation='relu', padding='valid'))
model.add(MaxPool1D(pool_size=2, strides=2))
model.add(Conv1D(10, kernel_size=2, strides=1, activation='relu', padding='valid'))
model.add(MaxPool1D(pool_size=2, strides=2))
model.add(Flatten())
model.add(Dense(units=1))

#模型编译
model.compile(optimizer='adam', loss='mse')

#模型训练
model.fit(x_train, y_train, epochs=300, batch_size=3)

#模型预测
norm_y_pred = model.predict(x_test)

#反归一化
y_pred = scaler.inverse_transform(norm_y_pred)

#绘制预测结果
plt.figure()
n = np.arange(len(data))
plt.plot(n, data, label='真实值')
plt.plot(n[-len(y_pred):], y_pred, label='预测值')
plt.xlabel('日期点/天')
plt.ylabel('申购金额/元')
plt.legend()
plt.show()

