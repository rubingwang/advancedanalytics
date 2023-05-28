import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import random

# 读取训练数据集
df_train = pd.read_csv('combine_train.csv')
df_train = df_train.fillna(df_train.median())

with open('df_names.txt', 'r') as f:
    cols_feature = f.read()

# 用逗号分隔的字符串，每个值加上双引号
cols_feature_quoted = ",".join(['"' + col.strip() + '"' for col in cols_feature.split(",")])

# 解析为Python列表
cols_feature = eval('[' + cols_feature_quoted + ']')
print(cols_feature)

X_train = df_train[cols_feature]
y_train = df_train['target']

# 将数据集分割成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=40)

# 构建神经网络模型
model = MLPRegressor(hidden_layer_sizes=(100,5), activation='relu', solver='sgd', max_iter=200)

# 训练模型
model.fit(X_train, y_train)

# 在测试集上测试模型
y_pred = model.predict(X_test)
y_pred = np.round(y_pred).astype(int)

score = model.score(X_test, y_test)

# 计算预测结果的评价指标
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R2) score:", r2)

# 保存模型
joblib.dump(model, 'model.pkl')



# 加载模型
model = joblib.load('model.pkl')

# 读取预测数据集
df_validation = pd.read_csv('combine_test.csv')
#df_validation[cols_feature] = df_validation[cols_feature].fillna(df_validation[cols_feature].median())

# 提取需要的特征
X_validation = df_validation[cols_feature]

# 预测目标变量
y_pred = model.predict(X_validation)
y_pred = np.round(y_pred).astype(int)

# 将预测结果保存到新的列中
df_validation['PRED'] = y_pred

# 更改列名
df_validation.rename(columns={'property_id': 'ID'}, inplace=True)

# 输出结果到csv文件
df_validation[['ID', 'PRED']].to_csv('prediction.csv', index=False)

