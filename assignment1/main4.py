import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import TweedieRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
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

#cols_feature = []
X_train = df_train[cols_feature]
y_train = df_train['target']

# 将数据集分割成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=40)

# 构建GLM模型
glm = TweedieRegressor(power=1, alpha=0.5, link='log')

# 构建标准化Pipeline
standard_pipeline = Pipeline(steps=[('standardize', StandardScaler())])

# 使用TransformedTargetRegressor将目标变量log变换，然后再进行标准化
model = TransformedTargetRegressor(regressor=standard_pipeline, transformer=np.log1p)

# 将模型放在Pipeline中，依次执行标准化、log变换和训练
pipeline = Pipeline(steps=[('standardize', standard_pipeline), ('log_transform', model), ('glm', glm)])
pipeline.fit(X_train, y_train)

# 在测试集上测试模型
y_pred = pipeline.predict(X_test)

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
joblib.dump(pipeline, 'model.pkl')
