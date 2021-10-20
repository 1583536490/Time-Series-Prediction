#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author: herry
@software: PyCharm
@file: BPNN
@time: 2021/10/7
"""
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers,optimizers
from sklearn.metrics import mean_squared_error as MSE

# 读取数据
def read_data(datapath):
    # read data
    data1 = pd.read_csv(datapath,index_col=0)
    return data1

# 分割数据
def splitdata(data,Sequence_length,horizon):
    all_data = []
    for dta in range(len(data) - Sequence_length -horizon + 1):
        all_data.append(data[dta : dta + Sequence_length])

    all_data = np.array(all_data)
    ydata = data[(horizon + Sequence_length -1) :]
    ydata = list(ydata)

    # 加入y变量
    all_data = pd.DataFrame(all_data)
    all_data['y'] = ydata
    all_data = np.array(all_data)

    # split data-训练数据、验证数据、测试数据比例0.6/0.2/0.2
    # round函数-进行四舍五入转化的
    row_1 =round(0.4 * int(all_data.shape[0]))
    row_2 = round(0.2 * int(all_data.shape[0]))

    x_train_initial = all_data[:-int(row_1),:-1]
    y_train_initial = all_data[:-int(row_1),-1]

    x_val_initial = all_data[-int(row_1):-int(row_2),:-1]
    y_val_initial = all_data[-int(row_1):-int(row_2),-1]

    x_test_initial = all_data[-int(row_2):,:-1]
    y_test_initial = all_data[-int(row_2):,-1]

    return x_train_initial,y_train_initial,x_val_initial,y_val_initial,x_test_initial,y_test_initial


# 标准化处理
def standard(x_train_initial,y_train_initial,x_val_initial,y_val_initial,x_test_initial,y_test_initial):
    MinMaxScaler


    x_scaler = StandardScaler()
    y_scale = StandardScaler()

    # 分别对x与y进行标准化
    x_train = x_scaler.fit_transform(x_train_initial)
    x_val = x_scaler.transform(x_val_initial)
    x_test = x_scaler.transform(x_test_initial)

    y_train = y_scale.fit_transform(y_train_initial.reshape(-1,1))
    y_val = y_scale.transform(y_val_initial.reshape(-1,1))
    y_test = y_scale.transform(y_test_initial.reshape(-1,1))

    # 生成正式的数据格式
    amount_of_features = 1
    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],amount_of_features))
    x_val = np.reshape(x_val,(x_val.shape[0],x_val.shape[1],amount_of_features))
    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],amount_of_features))

    return x_train,y_train,x_val,y_val,x_test,y_test

# keras 版本模型
def get_uncompiled_model():

    inputs = tf.keras.Input(shape=(60,))

    x = layers.Dense(60,activation='relu',name='dense_1')(inputs)
    x = layers.Dense(10,activation='relu',name='dense_2')(x)
    x = layers.Dense(1,name='dense_3')(x)

    model = tf.keras.Model(inputs=inputs,outputs=x)

    return model

# 转换到标准化前的格式
def inverse_data(y_train_initial,y_pre,y_test):
    y_scale = StandardScaler()
    y_scale = y_scale.fit(y_train_initial.reshape(-1,1))
    y_pre = y_scale.inverse_transform(y_pre)
    y_test = y_scale.inverse_transform(y_test)

    return y_pre,y_test

if __name__ == '__main__':
    # 读取数据-dataframe
    index = 'high'
    path = 'D:\python_study\study/shww1.csv'
    sequence_length = 60           # 序列长度
    horizon = 1                    # 移动的步长
    data = read_data(path)[index]
    # 分割数据
    x_train_initial,y_train_initial,x_val_initial,y_val_initial,x_test_initial,y_test_initial = splitdata(data,sequence_length,horizon=horizon)
    # 标准化
    x_train,y_train,x_val,y_val,x_test,y_test = standard(x_train_initial,y_train_initial,x_val_initial,y_val_initial,x_test_initial,y_test_initial)

    # print('size = ',y_test.size)

    # 训练模型
    model = get_uncompiled_model()
    # model.compile()模型配置损失和度量、优化器
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-4),
                  loss='mean_squared_error',
                 )
    # model.fit()对模型进行训练
    model.fit(x_train, y_train, batch_size=32, epochs=500,validation_data=(x_val,y_val))
    # model.predict()对模型进行预测
    y_pre = model.predict(x_test)
    # print('y_pre = ',y_pre)
    # model.summary()

    # 转换到原来的数据格式
    y_pre_rel,y_test_rel = inverse_data(y_train_initial,y_pre,y_test)

    y_pre_rel = np.array(y_pre_rel).reshape(-1,1)
    y_test_rel = np.array(y_test_rel).reshape(-1,1)

    # rmse
    rmse = np.sqrt(MSE(y_pre_rel,y_test_rel))

    # 画图
    # pd.Series()
    fig = plt.figure()
    plt.plot(y_pre_rel,'y-',label='predictions')
    plt.plot(y_test_rel,'r--',label='test')
    plt.legend(loc = 'best')
    plt.title('RMSE : %.4f'%(rmse))
    plt.show()
