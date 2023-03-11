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

train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')
types = train.dtypes

# =============================================================================
# deal with char column
# =============================================================================

char_features = ["property_name", "property_summary", "property_space", "property_desc", 
                 "property_neighborhood", "property_notes", "property_transit", 
                 "property_access", "property_interaction", "property_rules"]

train_text = train[char_features]

#合併文字欄位們 小寫
train_text["char_features"] = ( train["property_name"].astype(str) + " " + train["property_summary"].astype(str) + " " 
                               + train["property_space"].astype(str) + " " + train["property_desc"].astype(str) + " "
                               + train["property_neighborhood"].astype(str) + " " + train["property_notes"].astype(str) + " "
                               + train["property_transit"].astype(str) + " " + train["property_access"].astype(str) + " "
                               + train["property_interaction"].astype(str) + " " + train["property_rules"].astype(str) ).str.lower()

# 判斷語言
ch_ger = ["ä", "ö", "ß", " der ", " die ", " das "] # "ü" 法文也有
ch_fre = ["ç", "œu", "â", "ê", "î", "ô", "û", "ë", "ï", "ü", "è", "à", "ù", "é", " la ", " le ", " les ", " l'a", " l'e"] # "ü" 德文也有 
ch_dut = ["aa", "ij", " de ", " het ", " een ", " op "]
train_text["language"] = train_text.apply(lambda x: "German" if any(ch in x["char_features"] for ch in ch_ger) else 
                                          ("French" if any(ch in x["char_features"] for ch in ch_fre) else
                                           ("Dutch" if any(ch in x["char_features"] for ch in ch_dut) else 
                                            "English")), axis = 1).astype(object)
train_text.groupby(["language"]).size()

words = ""
for i in range(0,train_text.shape[0]):
    words = words + " " + train_text["char_features"][i]