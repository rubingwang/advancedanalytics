import pandas as pd
from sklearn.cluster import KMeans
from geopy.distance import geodesic

# Load data
df = pd.read_csv("train.csv", usecols=['property_id', 'property_rules', 'property_zipcode',
                                       'property_lat', 'property_lon', 'property_type',
                                       'property_room_type', 'property_max_guests',
                                       'property_bathrooms', 'property_bedrooms'])

# Code categorical columns
df['property_type'] = pd.Categorical(df['property_type']).codes
df['property_room_type'] = pd.Categorical(df['property_room_type']).codes

# Convert latitude and longitude into new two features columns
# Cluster by K=10, using latitude and longitude
coords = df[['property_lat', 'property_lon']]
kmeans = KMeans(n_clusters=10, random_state=42).fit(coords)
cluster_labels = kmeans.labels_
centroid_coords = kmeans.cluster_centers_

# Generate new feature columns 'cluster_label'
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

# Generate new feature columns 'centroid_distance'
df['centroid_distance'] = [round(x, 2) for x in centroid_distances]

# Drop rows with NA
df = df.drop(df[df[['property_zipcode', 'property_bathrooms', 'property_bedrooms']].isnull().any(axis=1)].index)

# Drop useless columns
df = df.drop(columns=['property_rules', 'property_lat', 'property_lon'], axis=1)

print(df.head())
print(df.isna().sum())

# Save into CSV file
df.to_csv('data_rubing.csv', index=False)
