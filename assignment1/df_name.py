import pandas as pd

# 读取数据帧（DataFrame）
df = pd.read_csv('combine_train_target.csv')

# 将变量名称加上双引号
quoted_columns = ['"' + col + '"' for col in df.columns]

# 将加上双引号的变量名称写入文件
with open('df_names.txt', 'w') as f:
    f.write(','.join(quoted_columns))
