import pandas as pd

# 读取数据帧（DataFrame）
df = pd.read_csv('combine_train_target.csv')

# 提取变量名称并写入文件
with open('df_names.txt', 'w') as f:
    f.write(','.join(df.columns))
