#This file is to clean part of raw data

from statistics import mean

import pandas as pd
from sklearn.cluster import KMeans
from geopy.distance import geodesic

# Load data
feature_cols=(['property_id', 'property_rules', 'property_zipcode',
               'property_lat', 'property_lon', 'property_type',
               'property_room_type', 'property_max_guests',
               'property_bathrooms', 'property_bedrooms'])

df1 = pd.read_csv("train.csv", usecols=feature_cols)
df2 = pd.read_csv("test.csv", usecols=feature_cols)


def process_data(df):
    # Code categorical columns
    for col in ['property_type', 'property_room_type']:
        df[col] = pd.Categorical(df[col]).codes

    # Cluster by K=2, using latitude and longitude
    coords = df[['property_lat', 'property_lon']]
    kmeans = KMeans(n_clusters=2, random_state=42).fit(coords)
    cluster_labels = kmeans.labels_
    centroid_coords = kmeans.cluster_centers_

    # Generate new feature columns 'cluster_label'
    df['cluster_label'] = cluster_labels.tolist()

    # Calculate distance between centroid of one cluster and property
    def compute_distance(lat, lon, centroid):
        return geodesic((lat, lon), centroid).km

    centroid_distances = []
    for i in range(len(df)):
        lat, lon = df.loc[i, ['property_lat', 'property_lon']]
        centroid = centroid_coords[cluster_labels[i]]
        distance = compute_distance(lat, lon, centroid)
        centroid_distances.append(distance)

    # Generate new feature columns 'centroid_distance'
    df['centroid_distance'] = [round(x, 2) for x in centroid_distances]

    # Fill NA
    #df = df.drop(df[df[['property_zipcode', 'property_bathrooms', 'property_bedrooms']].isnull().any(axis=1)].index)
    df[['property_bathrooms', 'property_bedrooms']] = \
        df[['property_bathrooms', 'property_bedrooms']].fillna(df[['property_bathrooms', 'property_bedrooms']].mode().iloc[0])

    # Drop useless columns
    df = df.drop(columns=['property_rules', 'property_zipcode', 'property_lat', 'property_lon'], axis=1)

    return df


# Process data
df3 = process_data(df1)
df4 = process_data(df2)

print(df3.head())
print(df3.isna().sum())
print(df3.dtypes)
print(df4.head())
print(df4.isna().sum())
print(df3.dtypes)

# Save into CSV file
df3.to_csv('rubing_train.csv', index=False)
df4.to_csv('rubing_test.csv', index=False)
