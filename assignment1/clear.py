# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 09:31:32 2023

@author: jjw13
"""
from sklearn.impute import KNNImputer
import pandas as pd
# 导入numpy库
import numpy as np
import datetime
import numpy as np






# 读取CSV文件，跳过第一行（标题行）
df = pd.read_csv('train.csv', skiprows=[0])

# 检查是否有重复的行
duplicates = df[df.duplicated(keep=False)]

# 输出重复行的行号和重合的行
if len(duplicates) > 0:
    print("df矩阵中存在以下重复的行：")
    duplicate_indices = []
    for i, row1 in duplicates.iterrows():
        for j, row2 in duplicates.iterrows():
            if i < j and (row1==row2).all():
                if i not in duplicate_indices:
                    duplicate_indices.append(i)
                if j not in duplicate_indices:
                    duplicate_indices.append(j)
    for index in duplicate_indices:
        duplicate_rows = df[(df == df.loc[index]).all(axis=1)]
        print(f"行号 {index} 与以下行号重合：{duplicate_rows.index.tolist()}")
else:
    print("df矩阵中没有重复的行")
    duplicate_indices = []

# 选择第37到45列
df_selected = df.iloc[:, 37:46]

# 输出所选列的数据
#print(df_selected)


# 检查是否有重复的行
duplicates = df_selected[df_selected.duplicated(keep=False)]

'''# 输出重复行的行号和重合的行
if len(duplicates) > 0:
    print("df_selected矩阵中存在以下重复的行：")
    duplicate_indices = []
    for i, row1 in duplicates.iterrows():
        for j, row2 in duplicates.iterrows():
            if i < j and (row1==row2).all():
                if i not in duplicate_indices:
                    duplicate_indices.append(i)
                if j not in duplicate_indices:
                    duplicate_indices.append(j)
    for index in duplicate_indices:
        duplicate_rows = df_selected[(df_selected == df_selected.loc[index]).all(axis=1)]
        print(f"行号 {index} 与以下行号重合：{duplicate_rows.index.tolist()}")
else:
    print("df_selected矩阵中没有重复的行")
    duplicate_indices = []
'''




################################################################判断数据中是否有缺省值或者不合法的数据
# 将前四列转换为列表
df_selected_columns = df_selected.iloc[:, :4].values.tolist()

# 将列表转换为Series类型
df_selected_columns = pd.Series([item for sublist in df_selected_columns for item in sublist])

# 判断前四列是否有非数值类型的元素
is_non_numeric = pd.to_numeric(df_selected_columns, errors='coerce').isnull().any()

# 输出结果
if is_non_numeric:
    print("前四列存在非数值类型的元素")
else:
    print("前四列全是数值类型的元素")



####################################################################################################################################
def set_new_col_value(row):
    if row[4] == "flexible":
        return 3
    elif row[4] == "moderate":
        return 2
    elif row[4] == "strict":
        return 1
    else:
        return 0

# 新增加一列，根据第五列数据设置新列的值
df_selected['new_col'] = df_selected.apply(lambda row: set_new_col_value(row), axis=1)
####################################################################################################################################







# 假设数据在df_selected中
# 将第七列和第八列的缺失值替换为0
df_selected.iloc[:, 6:8] = df_selected.iloc[:, 6:8].fillna(0)




# 假设数据在df_selected中
# 将第六列中所有非数值类型的数据改为缺省值，并报告缺省值所在的行号
df_selected.iloc[:, 5] = pd.to_numeric(df_selected.iloc[:, 5], errors='coerce')
missing_values = df_selected[df_selected.iloc[:, 5].isna()]
print("第六列变为缺省值的行号：")
print(missing_values.index.tolist())



# 假设数据在df_selected中
# 使用3-nearest方法填补第六列数据中的缺省值，使用前三列和第六列数据作为依据
imputer = KNNImputer(n_neighbors=3)
new_col = imputer.fit_transform(df_selected.iloc[:, [0, 1, 2, 5]])[:, 3]
df_selected.iloc[:, 5] = pd.Series(new_col, index=df_selected.index)




# 假设数据在df_selected中
# 使用3-nearest方法填补第九列数据中的缺省值，使用前三列和第九列数据作为依据
imputer = KNNImputer(n_neighbors=3)
new_col = imputer.fit_transform(df_selected.iloc[:, [0, 1, 2, 8]])[:, 3]
df_selected.iloc[:, 8] = pd.Series(new_col, index=df_selected.index)



# 假设数据在df_selected中
# 将第七列和第八列数据转换为日期数据类型
df_selected.iloc[:, 6:8] = df_selected.iloc[:, 6:8].apply(pd.to_datetime)



# 假设数据在df_selected中
# 将df_selected输出为Excel文件
df_selected.to_excel("output.xlsx", index=False)