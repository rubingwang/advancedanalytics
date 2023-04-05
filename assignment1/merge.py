import pandas as pd

# 创建示例数据
df1 = pd.read_csv('combine_train.csv')
df2 = pd.read_csv('train.csv')

# 将df2中的'target'合并到df1中
merged_df = pd.merge(df1, df2[['property_id','target']], on='property_id', how='left')

# 连接其他列
merged_df = merged_df.drop(df1.columns[0], axis=1)

merged_df.to_csv('combine_train_target.csv')

