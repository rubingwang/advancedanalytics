import pandas as pd

data = pd.read_csv("train.csv")
col_clean = ['property_id', 'property_rules', 'property_zipcode',
             'property_lat', 'property_lon', 'property_type',
             'property_room_type', 'property_max_guests',
             'property_bathrooms', 'property_bedrooms']

df = data[col_clean]

print(vars(df))
print(df.info())

print(df.isna().sum())

unique_values = df.apply(lambda x: x.unique())
print(unique_values)

from collections import Counter

for col in ['property_type', 'property_room_type']:
    word_counts = Counter(' '.join(df[col].astype(str)).split())
    print(f"{col} word counts:")
    print(word_counts)

from sklearn.cluster import KMeans
from geopy.distance import geodesic

# Cluster by K=10, using latitude and longitude
coords = df[['property_lat', 'property_lon']]
kmeans = KMeans(n_clusters=10, random_state=42).fit(coords)
cluster_labels = kmeans.labels_
centroid_coords = kmeans.cluster_centers_

# convert these two information into two new columns
df['cluster_label'] = cluster_labels.tolist()


# calculate distance between centroid of one cluster and property
def compute_distance(lat, lon, centroid):
    return geodesic((lat, lon), centroid).km


centroid_distances = []
for i in range(len(df)):
    lat, lon = df.loc[i, ['property_lat', 'property_lon']]
    centroid = centroid_coords[cluster_labels[i]]
    distance = compute_distance(lat, lon, centroid)
    centroid_distances.append(distance)

df['centroid_distance'] = [round(x, 2) for x in centroid_distances]

# 打印处理后的数据
print(df.head())


# 将数据转换为类别类型
df['property_room_type'] = pd.Categorical(df['property_room_type'])
df['property_type'] = pd.Categorical(df['property_type'])

# 将类别编码为整数
df['property_room_type'] = df['property_room_type'].cat.codes
df['property_type'] = df['property_type'].cat.codes

# 输出转换后的数据
print(df['property_room_type'])

# 使用 drop() 方法删除包含缺失值的行
df = df.drop(df[df[['property_zipcode', 'property_bathrooms', 'property_bedrooms']].isnull().any(axis=1)].index)

# 输出处理后的 DataFrame
print(df)

print(df.isna().sum())


# 使用 drop() 方法删除 property_rules 列
df = df.drop(columns=['property_lat', 'property_lon', 'property_rules'], axis=1)

# 输出处理后的 DataFrame
print(df)