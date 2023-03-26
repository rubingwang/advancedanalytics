import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import random

random.seed(42)

# 读取训练数据集
df_train = pd.read_csv('train_cleaned.csv')


# 提取需要的特征和目标变量
cols_feature = ['language', 'property_type', 'property_room_type',
                'property_max_guests', 'property_beds',
                '24-hourcheck-in', 'breakfast', 'booking_cancel_policy', 'reviews_num', 'reviews_rating', 'reviews_cleanliness']

X_train = df_train[cols_feature]
y_train = df_train['target']

# 将数据集分割成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# 构建神经网络模型
model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=1000)

# 训练模型
model.fit(X_train, y_train)

# 在测试集上测试模型
y_pred = model.predict(X_test)
score = model.score(X_test, y_test)

# 计算预测结果的评价指标
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R2-score:", r2)

# 保存模型
joblib.dump(model, 'model.pkl')


'''

# 加载模型
# model = joblib.load('model.pkl')

# 读取预测数据集
df_test = pd.read_csv('combine_test.csv')

# 提取需要的特征
X_test = df_test[['name1', 'name2', 'name3', 'name4', 'name5']]

# 预测目标变量
y_pred = model.predict(X_test)

# 将预测结果保存到新的列中
df_test['PRED'] = y_pred

# 更改列名
df_test.rename(columns={'property_id': 'ID'}, inplace=True)

# 输出结果到csv文件
df_test[['ID', 'PRED']].to_csv('prediction.csv', index=False)

'''