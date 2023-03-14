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
#customize possible missing value
missing_value_formats = ["n.a.","?","NA","n/a", "na", "--"]
train=pd.read_csv("/Users/bijiben/Desktop/BDA/advancedanalytics/assignment1/train.csv", na_values = missing_value_formats)
clear_list=['property_id', 'host_about', 'host_response_time', 'host_response_rate', 'host_nr_listings', 'host_nr_listings_total', 
      'host_verified', 'booking_price_covers', 'booking_min_nights', 'booking_max_nights', 'property_beds']

#investigate data type 
types=df.dtypes
print(types)

#check missingness in data
df.isnull().sum()
#quite lot null values in data, can not directly drop the row

#process numeric value
num_list=['host_response_rate','host_nr_listings',
          'host_nr_listings_total','booking_price_covers','booking_min_nights', 'booking_max_nights', 'property_beds']
num=df[num_list]
num.head(10)

#plot distribution of numeric features
num.hist(figsize=(10,20), bins=10);

#host_nr_listings and host_nr_listings_total
#since the outliers in the nr-listing
#replace missingness by median
median=df['host_nr_listings'].median
df['host_nr_listings'].fillna(median, inplace=True)
median_total=df['host_nr_listings_total'].median
df['host_nr_listings_total'].fillna(median_total, inplace=True)

#11 missingness
#hence, use k nereast neighbour imputation
from sklearn.impute import KNNImputer
imputer = KNNImputer()
df['property_beds']=imputer.fit_transform(df[['property_beds']])


#check missingness again
df.isnull().sum()


#there are 1461 missingness in host response time and host response rate
#can not directly ignore or impute with median or mode
#create a value represent missingness
print("Null values:", df.host_response_time.isna().sum())
print(f"Proportion: {round((df.host_response_time.isna().sum()/len(df))*100, 1)}%")

df.host_response_time.fillna("unknown", inplace=True)
df.host_response_time.value_counts(normalize=True)
#5 category in total

print("Mean host response rate:", round(df['host_response_rate'].mean(),0))
print("Median host response rate:", df['host_response_rate'].median())
print(f"Proportion of 100% host response rates: {round(((df.host_response_rate == 100.0).sum()/df.host_response_rate.count())*100,1)}%")

#according to histogram, convert into four categories for better interpretation
df.host_response_rate = pd.cut(df.host_response_rate,
                               bins=[0, 50, 90, 99, 100], labels=['0-49%', '50-89%', '90-99%', '100%'], include_lowest=True)

#encode host_response time
#unkonw is encoded as 1
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

#create new column indicating possible addition bed
df['available_extra_beds']=df['property_beds']-df['booking_price_covers']

#drop unnecessary columns：host_about, host_verified(since already convert to binary variable)
df=df.drop(['host_about', 'host_verified'], axis=1)

#export clean data
df.to_csv('siwen.csv')