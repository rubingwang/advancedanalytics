
import pandas as pd


df = pd.read_csv('assignment1/train_cleaned.csv')

df['language'] = df['language'].astype('category')
df['language'] = df['language'].cat.codes

#print(df.isna().sum())

# 检查每一列是否存在NA值
na_cols = df.isna().any()
# 打印存在NA值的列名
for col in na_cols[na_cols == True].index:
    print(col)

#df.to_csv('assignment1/train_cleaned.csv')

