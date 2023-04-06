import pandas as pd

# 读取数据
df = pd.read_csv('combine_train_target.csv')
df = df.drop(df.columns[[127, 128]], axis=1)

# 查找包含'2017-05-09'的列名
cols_containing_20170509 = df.columns[df.columns.str.contains('2017-05-09')]

print(cols_containing_20170509)
