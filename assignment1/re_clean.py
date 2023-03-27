''
import pandas as pd
'''
df = pd.read_csv('train_cleaned.csv')

df['language'] = df['language'].astype('category')
df['language'] = df['language'].cat.codes

#print(df.isna().sum())

# 检查每一列是否存在NA值
na_cols = df.isna().any()
# 打印存在NA值的列名
for col in na_cols[na_cols == True].index:
    print(col)


'''
#df.to_csv('train_cleaned.csv')

df = pd.read_csv('train_cleaned_filled.csv')
#打印变量名称
df_columns = list(df.columns)
df_columns_quoted = ["'" + name + "'," for name in df.columns]
with open('df_names.txt', 'w') as f:
    for name in df_columns_quoted:
        f.write("%s\n" % name)




'''
import pandas as pd

# 读取数据集
df = pd.read_csv('train_cleaned.csv')

# 计算每个变量的缺失值数量
missing_values_count = df.isnull().sum()

# 按照缺失值数量从大到小进行排序
missing_values_count_sorted = missing_values_count.sort_values(ascending=False)

# 打印排序后的结果
print(missing_values_count_sorted)
'''
'''
import pandas as pd

# 读取数据集
df = pd.read_csv('train_cleaned.csv')

# 删除名为 scraped_minus_review 的列
df = df.drop(['scraped_minus_review','host_since','scraped_at'], axis=1)

# 检查每个变量是否有缺失值
for col in df.columns:
    if df[col].isnull().values.any():
        # 如果存在缺失值，则填充缺失值
        df[col].fillna(df[col].mean(), inplace=True)

# 保存处理后的数据集
df.to_csv('train_cleaned_filled.csv', index=False)
'''