import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Load data
df = pd.read_csv('train_cleaned.csv')

# Select features from cols_feature variable
with open('df_names.txt', 'r') as f:
    df_columns_quoted = f.read()

# 移除最后一个逗号
df_columns_quoted = df_columns_quoted[:-2]

# 解释为 Python 代码，并赋值给变量 df_columns
cols_feature = eval('[' + df_columns_quoted + ']')

best_features = []
min_rmse = float('inf')

for i in range(1):
    # 从cols_feature中随机抽取20个feature训练
    features = random.sample(cols_feature, 20)

    #cols_feature = []
    X_train = df_train[cols_feature]
    y_train = df_train['target']

    # 将数据集分割成训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=40)


    # 构建神经网络模型
    model = MLPRegressor(hidden_layer_sizes=(30,), activation='relu', solver='lbfgs')

    # 训练模型
    model.fit(X_train_20, y_train)

    # 在测试集上测试模型
    y_pred = model.predict(X_test_20)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    if rmse < min_rmse:
        min_rmse = rmse
        best_features = features

    print(f"RMSE of iteration {i + 1}: {rmse}")

print("Best features:", best_features)
print("Min RMSE:", min_rmse)
