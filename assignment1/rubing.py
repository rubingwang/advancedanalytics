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

    # Cluster by K=10, using latitude and longitude
    coords = df[['property_lat', 'property_lon']]
    kmeans = KMeans(n_clusters=10, random_state=42).fit(coords)
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

    # Drop rows with NA
    df = df.drop(df[df[['property_zipcode', 'property_bathrooms', 'property_bedrooms']].isnull().any(axis=1)].index)

    # Drop useless columns
    df = df.drop(columns=['property_rules', 'property_zipcode', 'property_lat', 'property_lon'], axis=1)

    return df


# Process data
df1 = process_data(df1)
df2 = process_data(df2)

print(df1.head())
print(df1.isna().sum())
print(df2.head())
print(df2.isna().sum())

# Save into CSV file
df1.to_csv('train_rubing.csv', index=False)
df2.to_csv('test_rubing.csv', index=False)