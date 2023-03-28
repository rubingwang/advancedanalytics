import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import string
from sklearn.preprocessing import OrdinalEncoder
import nltk
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

pd.options.mode.chained_assignment = None  # default='warn'

missing_value_formats = ["n.a.","?","NA","n/a", "na", "--"]
test=pd.read_csv("/Users/bijiben/Desktop/BDA/advancedanalytics/assignment1/test.csv", na_values = missing_value_formats)
clear_list=['property_id', 'host_about', 'host_response_time', 'host_response_rate', 'host_nr_listings', 'host_nr_listings_total', 
      'host_verified', 'booking_price_covers', 'booking_min_nights', 'booking_max_nights', 'property_beds']
df=test[clear_list]
#investigate data type 
types=df.dtypes
print(types)


#check missingness in data
df.isnull().sum()

df['host_nr_listings'] = df['host_nr_listings'].fillna(df['host_nr_listings'].median())
df['host_nr_listings_total'] = df['host_nr_listings_total'].fillna(df['host_nr_listings_total'].median())

from sklearn.impute import KNNImputer
imputer = KNNImputer()
df['property_beds']=imputer.fit_transform(df[['property_beds']])

df.isnull().sum()

df.host_response_time.fillna("unknown", inplace=True)
df.host_response_time.value_counts(normalize=True)
ord_enc = OrdinalEncoder(categories=[['unknown', 'within an hour', 'within a few hours', 'within a day', 'a few days or more']])
df["responsetime_code"] = ord_enc.fit_transform(df[["host_response_time"]])
df[["host_response_time", "responsetime_code"]].head(20)

imputer = KNNImputer()
df['host_response_rate']=imputer.fit_transform(df[['host_response_rate']])

#process host_verified
df.host_verified.value_counts()

#convert verifying document with and without official id into booleen and binary data
df['logic_host_verified']= df.host_verified.str.contains("offline_government_id|government_id|driver’s license|passport|identity card|id|visa", regex=True)
df['logic_host_verified'].head(10)

test=pd.read_csv("/Users/bijiben/Desktop/BDA/advancedanalytics/assignment1/combine_test.csv", engine='python', dtype={'user_id': int})
test=test[['property_id', 'property_max_guests']]
df=pd.merge(df, test, on='property_id')

df['extra_beds']=df['property_max_guests']-df['booking_price_covers']
df['extra_beds'].head()

df.isnull().sum()

df=df.drop(['host_about', 'host_verified', 'property_max_guests','host_response_time'], axis=1)
df.head(5)
types=df.dtypes
print(types)

df.to_csv('siwen_test.csv')





