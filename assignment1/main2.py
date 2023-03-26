import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import joblib
import random

# 读取训练数据集
df_train = pd.read_csv('train_cleaned_filled.csv')

# 提取需要的特征和目标变量
with open('df_names.txt', 'r') as f:
    df_columns_quoted = f.read()

# 移除最后一个逗号
df_columns_quoted = df_columns_quoted[:-2]

# 解释为 Python 代码，并赋值给变量 df_columns
cols_feature = eval('[' + df_columns_quoted + ']')

#cols_feature = []

# 将数据集转换成 Numpy 数组，并进行归一化处理
X_train = df_train[cols_feature].values
y_train = df_train['target'].values.reshape(-1, 1)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

# 构建 Autoencoder 模型
input_dim = X_train.shape[1]
encoding_dim = 10  # 设置压缩后的维度为10

input_layer = Input(shape=(input_dim, ))
encoder_layer = Dense(encoding_dim, activation="relu")(input_layer)
decoder_layer = Dense(input_dim, activation="sigmoid")(encoder_layer)

autoencoder = Model(inputs=input_layer, outputs=decoder_layer)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# 训练 Autoencoder 模型
autoencoder.fit(X_train, X_train, epochs=100, batch_size=32)

# 获取 Autoencoder 的编码器
encoder = Model(inputs=input_layer, outputs=encoder_layer)

# 使用编码器对训练集进行压缩
X_train_compressed = encoder.predict(X_train)

# 将压缩后的数据作为特征
X_train = X_train_compressed

# 将数据集分割成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=40)

# 构建神经网络模型
model = MLPRegressor(hidden_layer_sizes=(20,), activation='relu', solver='lbfgs')

# 训练模型
model.fit(X_train, y_train)

# 在测试集上测试模型
y_pred = model.predict(X_test)
score = model.score(X_test, y_test)

# 计算预测结果的评价指标
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)

# 保存模型
#joblib.dump(model, 'model.pkl')

