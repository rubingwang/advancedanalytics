import pandas as pd
import numpy as np
import random
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.externals import joblib

# 读取训练数据集
df_train = pd.read_csv('train_cleaned_filled.csv')

# 提取需要的特征和目标变量
with open('df_names.txt', 'r') as f:
    df_columns_quoted = f.read()

# 移除最后一个逗号
df_columns_quoted = df_columns_quoted[:-2]

# 解释为 Python 代码，并赋值给变量 df_columns
cols_feature = eval('[' + df_columns_quoted + ']')

best_hidden_layer_sizes = ()
min_rmse = float('inf')

for i in range(100):
    # 从cols_feature中随机抽取20个feature训练
    features = random.sample(cols_feature, 20)

    # cols_feature = []
    X_train = df_train[cols_feature]
    y_train = df_train['target']

    # 将数据集分割成训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=40)

    # 随机选择隐藏层的大小进行训练
    hidden_layer_sizes = tuple([random.randint(10, 50) for i in range(random.randint(1, 3))])

    # 构建神经网络模型
    model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation='relu', solver='lbfgs')

    # 训练模型
    model.fit(X_train, y_train)

    # 在测试集上测试模型
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    if rmse < min_rmse:
        min_rmse = rmse
        best_hidden_layer_sizes = hidden_layer_sizes

    print(f"RMSE of iteration {i + 1}: {rmse}")

print("Best hidden_layer_sizes:", best_hidden_layer_sizes)
print("Min RMSE:", min_rmse)

# 用最优的隐藏层大小训练模型
model = MLPRegressor(hidden_layer_sizes=best_hidden_layer_sizes, activation='relu', solver='lbfgs')
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
joblib.dump(model, 'model.pkl')
