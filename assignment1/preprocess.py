# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 17:55:46 2023

@author: Penny
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import operator

os.chdir("C:/Users/Penny/Documents/魯汶/下學期/進階大數據分析/hw/hw1/advancedanalytics/assignment1")

# =============================================================================
# import data
# =============================================================================

train = pd.read_csv('train.csv', index_col= "property_id")
test  = pd.read_csv('test.csv', index_col= "property_id")
train_id = train.index.tolist()
test_id = test.index.tolist()
# train_id = train["property_id"].tolist()
# test_id = test["property_id"].tolist()
train_test = pd.concat([train, test], axis = 0)
types = train.dtypes
train.info()
test.info()

# =============================================================================
# deal with char column
# =============================================================================

char_features = ["property_name", "property_summary", "property_space", "property_desc", 
                 "property_neighborhood", "property_notes", "property_transit", 
                 "property_access", "property_interaction", "property_rules"]

# train_text = train[char_features]
# train_test_text = train_test[char_features]
train_test_text = pd.DataFrame({"property_id": train_test.index.tolist()})
train_test_text["property_id"] = train_test_text["property_id"].astype(object)
train_test_text = train_test_text.set_index("property_id")
# train_test_text = train_test["property_id"]
train_test_text = pd.concat([train_test_text, train_test[char_features]], axis = 1)
train_test_text.info()

#合併文字欄位們 小寫
# train_text["char_features"] = ( train["property_name"].astype(str) + " " + train["property_summary"].astype(str) + " " 
#                                + train["property_space"].astype(str) + " " + train["property_desc"].astype(str) + " "
#                                + train["property_neighborhood"].astype(str) + " " + train["property_notes"].astype(str) + " "
#                                + train["property_transit"].astype(str) + " " + train["property_access"].astype(str) + " "
#                                + train["property_interaction"].astype(str) + " " + train["property_rules"].astype(str) ).str.lower()
train_test_text["char_features"] = ( train_test["property_name"].astype(str) + " " + train_test["property_summary"].astype(str) + " " 
                               + train_test["property_space"].astype(str) + " " + train_test["property_desc"].astype(str) + " "
                               + train_test["property_neighborhood"].astype(str) + " " + train_test["property_notes"].astype(str) + " "
                               + train_test["property_transit"].astype(str) + " " + train_test["property_access"].astype(str) + " "
                               + train_test["property_interaction"].astype(str) + " " + train_test["property_rules"].astype(str) ).str.lower()

# 判斷語言
ch_ger = ["ä", "ö", "ß", " der ", " die ", " das "] # "ü" 法文也有
ch_fre = ["ç", "œu", "â", "ê", "î", "ô", "û", "ë", "ï", "ü", "è", "à", "ù", "é", " la ", " le ", " les ", " l'a", " l'e"] # "ü" 德文也有 
ch_dut = ["aa", "ij", " de ", " het ", " een ", " op "]
# train_text["language"] = train_text.apply(lambda x: "German" if any(ch in x["char_features"] for ch in ch_ger) else 
#                                           ("French" if any(ch in x["char_features"] for ch in ch_fre) else
#                                            ("Dutch" if any(ch in x["char_features"] for ch in ch_dut) else 
#                                             "English")), axis = 1).astype(object)
# train_text.groupby(["language"]).size()
train_test_text["language"] = train_test_text.apply(lambda x: "German" if any(ch in x["char_features"] for ch in ch_ger) else 
                                          ("French" if any(ch in x["char_features"] for ch in ch_fre) else
                                           ("Dutch" if any(ch in x["char_features"] for ch in ch_dut) else 
                                            "English")), axis = 1).astype(object)
train_test_text.groupby(["language"]).size()


# # one hot encoding
# from sklearn.preprocessing import OneHotEncoder

# #creating instance of one-hot-encoder
# encoder = OneHotEncoder(handle_unknown='ignore')

# #perform one-hot encoding on 'team' column 
# encoder_df = pd.DataFrame(encoder.fit_transform(train_test_text[['language']]).toarray())

# #merge one-hot encoded columns back with original DataFrame
# final_df = df.join(encoder_df)


# 匯出使用語言的欄位成CSV
train_text = pd.DataFrame({"property_id": train_id})
train_text = train_text.set_index("property_id")
train_text["language"] = train_test_text.loc[train_id, "language"]

test_text = pd.DataFrame({"property_id": test_id})
test_text = test_text.set_index("property_id")
test_text["language"] = train_test_text.loc[test_id, "language"]

train_text.to_csv('peiling_train.csv', header = True) 
test_text.to_csv('peiling_test.csv', header = True) 


# 文字雲
words = ""
words_drop = words
for i in range(0,train_test_text.shape[0]):
    words = words + " " + train_test_text["char_features"][i]
    
for i in [",", ".", "!"]:
    words_drop = words_drop.replace(i, " ")
    
for i in ["  ", "   "]:
    words_drop = words_drop.replace(i, " ")
    
for i in [" nan ", " la ", " de ", " le ", " et ", " du ", " un ", " en ", "de "]:
    words_drop = words_drop.replace(i, " ")
    
for i in ["brussel", "place", "my", "room"]:
    words_drop = words_drop.replace(i, "")
    
with open("word.txt", 'w', encoding="utf-8") as f:
    f.write(words)
with open("word_drop.txt", 'w', encoding="utf-8") as f:
    f.write(words_drop)
    
    
# =============================================================================
# combine datasets
# =============================================================================

train_1 = pd.read_csv('train_1.csv') #, index_col=("property_id"))
train_2 = pd.read_csv('train_2.csv') #, index_col=("property_id"))
train_3 = pd.read_csv('train_3.csv') #, index_col=("property_id"))
train_4 = pd.read_csv('train_4.csv') #, index_col=("property_id"))
train_5 = pd.read_csv('train_5.csv') #, index_col=("property_id"))
train_6 = pd.read_csv('train_6.csv') #, index_col=("property_id"))

train_y = pd.DataFrame({"property_id": train_id})
train_y["target"] = train["target"]
# train_y = train_y.set_index("property_id")

test_1 = pd.read_csv('test_1.csv') #, index_col=("property_id"))
test_2 = pd.read_csv('test_2.csv') #, index_col=("property_id"))
test_3 = pd.read_csv('test_3.csv') #, index_col=("property_id"))
test_4 = pd.read_csv('test_4.csv') #, index_col=("property_id"))
test_5 = pd.read_csv('test_5.csv') #, index_col=("property_id"))
test_6 = pd.read_csv('test_6.csv') #, index_col=("property_id"))

combine_train = pd.DataFrame({"property_id": train_id})
for i in [train_1, train_2, train_3, train_4, train_6, train_y]: #train_5 # [train_1, train_2, train_3, train_4, train_5, train_6]
    if "Unnamed: 0" in i.columns.tolist():
        print("drop")
        i = i.drop(["Unnamed: 0"], axis = 1)
    # i.reset_index(drop=True, inplace=True)      
    # combine_train = pd.concat([combine_train, i], axis = 1)
    combine_train = pd.merge(combine_train, i, on="property_id")
    

combine_test = pd.DataFrame({"property_id": test_id})
for i in [test_1, test_2, test_3, test_4, test_6]: # test_5,
    if "Unnamed: 0" in i.columns.tolist():
        print("drop")
        i = i.drop(["Unnamed: 0"], axis = 1)
    # i.reset_index(drop=True, inplace=True)    
    # combine_test = pd.concat([combine_test, i], axis = 1)
    combine_test = pd.merge(combine_test, i, on="property_id")

combine_train.to_csv('combine_train.csv', header = True, index = False) 
combine_test.to_csv('combine_test.csv', header = True, index = False) 














