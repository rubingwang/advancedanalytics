import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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



# 选取连续数据变量并标准化
# 从训练集中选取所有的连续型变量
continuous_cols = []
for col in df_train[cols_feature].select_dtypes(include=['float64', 'int64']).columns:
   if df_train[col].max() - df_train[col].min() > 10:
       continuous_cols.append(col)

continuous_cols = [col for col in continuous_cols if col != 'property_type']

#continuous_cols = df_train[cols_feature].select_dtypes(include=['float64', 'int64']).columns.tolist()
df_unstan = df_train[cols_feature].drop(continuous_cols, axis=1)
X_train_continuous = df_train[continuous_cols]
scaler = StandardScaler()
X_train_continuous = scaler.fit_transform(X_train_continuous)
X_train_continuous = pd.DataFrame(X_train_continuous, columns=continuous_cols)
print(continuous_cols)

# 将标准化后的连续数据和其他数据合并
X_train = pd.concat([X_train_continuous, df_unstan], axis=1)




#cols_feature = []
#X_train = df_train[cols_feature]
y_train = df_train['target']

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
r2 = r2_score(y_test, y_pred)

print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R2) score:", r2)

# 保存模型
joblib.dump(model, 'model.pkl')




'''
# 加载模型
# model = joblib.load('model.pkl')

# 读取预测数据集
df_test = pd.read_csv('combine_test.csv')

# 提取需要的特征
X_test = df_test[cols_feature]

# 预测目标变量
y_pred = model.predict(X_test)

# 将预测结果保存到新的列中
df_test['PRED'] = y_pred

# 更改列名
df_test.rename(columns={'property_id': 'ID'}, inplace=True)

# 输出结果到csv文件
df_test[['ID', 'PRED']].to_csv('prediction.csv', index=False)

'''