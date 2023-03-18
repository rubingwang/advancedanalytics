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

num_list=['host_response_rate','host_nr_listings',
          'host_nr_listings_total','booking_price_covers','booking_min_nights', 'booking_max_nights', 'property_beds']
num=df[num_list]
num.head(10)

num.hist(figsize=(10,20), bins=10);

median=df['host_nr_listings'].median
df['host_nr_listings'].fillna(median, inplace=True)
median_total=df['host_nr_listings_total'].median
df['host_nr_listings_total'].fillna(median_total, inplace=True)

from sklearn.impute import KNNImputer
imputer = KNNImputer()
df['property_beds']=imputer.fit_transform(df[['property_beds']])

df.isnull().sum()

df.host_response_time.fillna("unknown", inplace=True)
df.host_response_time.value_counts(normalize=True)

df.host_response_rate = pd.cut(df.host_response_rate,
                               bins=[0, 50, 90, 99, 100], labels=['0-49%', '50-89%', '90-99%', '100%'], include_lowest=True)
df.host_response_rate = df.host_response_rate.astype('str')
df.host_response_rate.replace('nan', 'unknown', inplace=True)
df.host_response_rate.value_counts()

ord_enc = OrdinalEncoder()
df["responsetime_code"] = ord_enc.fit_transform(df[["host_response_time"]])
df[["host_response_time", "responsetime_code"]].head(20)

#process host_verified
df.host_verified.value_counts()

#convert verifying document with and without official id into booleen and binary data
df['logit_host_verified']= df.host_verified.str.contains("offline_government_id|government_id|driver’s license|passport|identity card|id|visa", regex=True)
df['logit_host_verified'].head(10)
df['bin_host_verified']=df['logit_host_verified'].astype(int)
df['bin_host_verified'].head(10)

df['available_extra_beds']=df['property_beds']-df['booking_price_covers']

df=df.drop(['host_about', 'host_verified'], axis=1)

df.to_csv('test_clean.csv')





