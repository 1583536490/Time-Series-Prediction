from keras.layers import LSTM, Bidirectional
from keras.layers import merge
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import copy

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

SINGLE_ATTENTION_VECTOR = False


def attention_3d_block(inputs):
    input_dim = int(inputs.shape[2])
    a = inputs
    a = Dense(input_dim, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((1, 2), name='attention_vec')(a)

    output_attention_mul = merge.multiply([inputs, a_probs])
    return output_attention_mul


def attention_model():
    inputs = Input(shape=(TIME_STEPS, INPUT_DIMS))

    lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True, input_shape=(train_x.shape[1], 1)))(inputs)
    lstm_out = Dropout(0.3)(lstm_out)

    attention_mul = attention_3d_block(lstm_out)
    attention_mul = Flatten()(attention_mul)

    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(inputs=[inputs], outputs=output)
    return model


def create_new_dataset(dataset, seq_len):
    X = []  # 初始特征数据集为空列表
    y = []  # 初始标签数据集为空列表

    start = 0  # 初始位置
    end = dataset.shape[0] - seq_len  # 截止位置

    for i in range(start, end):  # for循环构造特征数据集
        sample = dataset[i: i + seq_len]  # 基于时间跨度seq_len创建样本
        label = dataset[i + seq_len]  # 创建sample对应的标签
        X.append(sample)  # 保存sample
        y.append(label)  # 保存label

    # 返回特征数据集和标签集
    return np.array(X), np.array(y)


def normalization(data, min, max):
    range = max - min
    m = data.shape[0]
    normData = data - np.tile(min, (1, m))
    normData = normData / np.tile(range, (1, m))
    return normData


# 加载数据
data = pd.read_csv("./MT1_2.csv", usecols=[0], engine='python')
data = data.values

len_data = len(data)
INPUT_DIMS = 1
TIME_STEPS = 24
lstm_units = 128
train_data_len = int(len_data * 0.8)

# 数据集划分
train_set = data[: train_data_len]  # 训练集
test_set = data[train_data_len - TIME_STEPS:]

# 归一化
scaler = MinMaxScaler(feature_range=(0, 1))
train_norm_1 = scaler.fit_transform(train_set)
max = scaler.data_max_
min = scaler.data_min_

test_norm_1 = copy.deepcopy(test_set)
for i in range(len(test_set)):
    tmp = test_norm_1[i]
    test_norm_1[i] = normalization(tmp, min, max)

train_x, train_y = create_new_dataset(train_norm_1, TIME_STEPS)
test_x, test_y = create_new_dataset(test_norm_1, TIME_STEPS)
test_x_, test_y_ = create_new_dataset(test_set, TIME_STEPS)

# 训练&预测
model = attention_model()
model.compile(optimizer='adam', loss='mse')
model.fit(train_x, train_y, epochs=150, batch_size=32, validation_split=0.1,shuffle=True)
test_pred_2 = model.predict(test_x)

# 反归一化
true_predictions = scaler.inverse_transform(np.array(test_pred_2).reshape(-1, 1))

# 度量指标数据
score_2 = r2_score(test_y_, true_predictions)
MAE_2 = mean_absolute_error(test_y_, true_predictions)
MSE_2 = mean_squared_error(test_y_, true_predictions)
RMSE_2 = math.sqrt(MSE_2)
print('MAE_1：', MAE_2)
print('MSE_1:', MSE_2)
print('RMSE_1:', RMSE_2)
print("r^2_1 的值:", score_2)

# 绘制图形
len_data = len(data)
x1 = np.arange(0, len_data - train_data_len, 1)

plt.figure(2)
plt.plot(x1, test_y_)
plt.plot(x1, true_predictions)
plt.show()
