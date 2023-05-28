import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 
from sklearn.model_selection import GridSearchCV, train_test_split
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error 

train=pd.read_csv("/Users/bijiben/Desktop/BDA/advancedanalytics/assignment1/combine_train.csv", engine='python')
train.head()

#since host_nr_listing has perfect correlation with host_total_nr_listing
#drop one f them
train=train.drop('host_nr_listings_total', axis=1)

from sklearn.impute import KNNImputer
imputer = KNNImputer()
train['target']=imputer.fit_transform(train[['target']])

y=train['target']
x=train.drop('target',axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print(y)

y.isnull().sum()

#normalize features
ss=StandardScaler()
x_trains=ss.fit_transform(x_train)
x_tests=ss.fit_transform(x_test)

#feature selection
svr=SVR()
svr.fit(x_trains,y_train)

#Recursive Feature selection using SVM
x_trains_df=pd.DataFrame(x_trains,columns=x_train.columns)
from sklearn.feature_selection import RFE
svr_lin=SVR(kernel='linear')
svm_rfe_model=RFE(estimator=svr_lin)
svm_rfe_model_fit=svm_rfe_model.fit(x_trains_df,y_train)
feat_index = pd.Series(data = svm_rfe_model_fit.ranking_, index = x_train.columns)
signi_feat_rfe = feat_index[feat_index==1].index
print('Significant features from RFE',signi_feat_rfe)

print('Original number of features present in the dataset : {}'.format(train.shape[1]))
print()
print('Number of features selected by the Recursive feature selection technique is : {}'.format(len(signi_feat_rfe)))

x_train_new=x_train[['property_id', 'English', 'property_max_guests', 'property_bedrooms',
       'cluster_label', 'accessible-heightbed', 'bbqgrill', 'breakfast',
       'buzzer/wirelessintercom', 'cabletv', 'cat(s)', 'changingtable',
       'children’sbooksandtoys', 'children’sdinnerware',
       'cleaningbeforecheckout', 'cookingbasics', 'disabledparkingspot',
       'dishesandsilverware', 'dishwasher', 'doorman', 'doormanentry', 'dryer',
       'elevatorinbuilding', 'essentials', 'ethernetconnection',
       'extrapillowsandblankets', 'family/kidfriendly', 'fireextinguisher',
       'firmmatress', 'firstaidkit', 'freeparkingonpremises',
       'gardenorbackyard', 'grab-railsforshowerandtoilet', 'heating',
       'indoorfireplace', 'iron', 'keypad', 'kitchen',
       'laptopfriendlyworkspace', 'lockbox', 'oven', 'pack’nplay/travelcrib',
       'patioorbalcony', 'pocketwifi', 'pool', 'privateentrance',
       'privatelivingroom', 'refrigerator',
       'roll-inshowerwithshowerbenchorchair', 'selfcheck-in',
       'singlelevelhome', 'smartlock', 'stairgates', 'stove',
       'translationmissing:en.hosting_amenity_49',
       'translationmissing:en.hosting_amenity_50', 'tv', 'widedoorway',
       'widehallwayclearance', 'property_last_updated', 'host_nr_listings',
       'booking_max_nights', 'property_beds', 'responsetime_code',
       'extra_beds', 'reviews_acc', 'reviews_cleanliness', 'reviews_checkin',
       'reviews_location', 'Host Is Superhost', 'Is Location Exact',
       'Require Guest Phone Verification', 'booking_availability_30',
       'booking_availability_60', 'reviews_num', 
       'reviews_rating', 'booking_cancel_policy']]
x_test_new=x_test[['property_id', 'English', 'property_max_guests', 'property_bedrooms',
       'cluster_label', 'accessible-heightbed', 'bbqgrill', 'breakfast',
       'buzzer/wirelessintercom', 'cabletv', 'cat(s)', 'changingtable',
       'children’sbooksandtoys', 'children’sdinnerware',
       'cleaningbeforecheckout', 'cookingbasics', 'disabledparkingspot',
       'dishesandsilverware', 'dishwasher', 'doorman', 'doormanentry', 'dryer',
       'elevatorinbuilding', 'essentials', 'ethernetconnection',
       'extrapillowsandblankets', 'family/kidfriendly', 'fireextinguisher',
       'firmmatress', 'firstaidkit', 'freeparkingonpremises',
       'gardenorbackyard', 'grab-railsforshowerandtoilet', 'heating',
       'indoorfireplace', 'iron', 'keypad', 'kitchen',
       'laptopfriendlyworkspace', 'lockbox', 'oven', 'pack’nplay/travelcrib',
       'patioorbalcony', 'pocketwifi', 'pool', 'privateentrance',
       'privatelivingroom', 'refrigerator',
       'roll-inshowerwithshowerbenchorchair', 'selfcheck-in',
       'singlelevelhome', 'smartlock', 'stairgates', 'stove',
       'translationmissing:en.hosting_amenity_49',
       'translationmissing:en.hosting_amenity_50', 'tv', 'widedoorway',
       'widehallwayclearance', 'property_last_updated', 'host_nr_listings',
       'booking_max_nights', 'property_beds', 'responsetime_code',
       'extra_beds', 'reviews_acc', 'reviews_cleanliness', 'reviews_checkin',
       'reviews_location', 'Host Is Superhost', 'Is Location Exact',
       'Require Guest Phone Verification', 'booking_availability_30',
       'booking_availability_60', 'reviews_num',
       'reviews_rating', 'booking_cancel_policy']]
#normalize features
ss=StandardScaler()
x_trains_new=ss.fit_transform(x_train_new)
x_tests_new=ss.fit_transform(x_test_new)

#build model with selected features
rfe_svm=SVR(kernel='linear')
rfe_fit=rfe_svm.fit(x_trains_new, y_train)
y_pred=rfe_fit.predict(x_tests_new)
rmse = float(format(np.sqrt(mean_squared_error(y_test, y_pred)), '.3f'))
print("\nRMSE: ", rmse)

from sklearn.model_selection import RandomizedSearchCV
kernels = list(['linear', 'rbf', 'poly', 'sigmoid'])
c = list([1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 1e2, 1e3])
gammas = list([0.1, 1, 10, 100])

ref = SVR()
ref.fit(x_trains_new, y_train)
param_random = dict(kernel=kernels, C=c, gamma=gammas)
random = RandomizedSearchCV(ref, param_random, cv=10, n_jobs=-1, verbose=3)
random.fit(x_trains_new, y_train)
random.best_params_

#use tuned hyper-parameter to fit model
model = SVR(kernel='rbf',gamma=1, C=10,  verbose=1,max_iter=2500).fit(x_trains_new, y_train)
y_pred = model.predict(x_tests_new)
rmse = float(format(np.sqrt(mean_squared_error(y_test, y_pred)), '.3f'))
print("\nRMSE: ", rmse)

ss=StandardScaler()
x=ss.fit_transform(x)


test=pd.read_csv("/Users/bijiben/Desktop/BDA/advancedanalytics/assignment1/combine_test.csv")
test.head()


print(test['reviews_last'])
test.drop('reviews_last', axis=1)

test_do=test[['property_id', 'English', 'property_max_guests', 'property_bedrooms',
       'cluster_label', 'accessible-heightbed', 'bbqgrill', 'breakfast',
       'buzzer/wirelessintercom', 'cabletv', 'cat(s)', 'changingtable',
       'children’sbooksandtoys', 'children’sdinnerware',
       'cleaningbeforecheckout', 'cookingbasics', 'disabledparkingspot',
       'dishesandsilverware', 'dishwasher', 'doorman', 'doormanentry', 'dryer',
       'elevatorinbuilding', 'essentials', 'ethernetconnection',
       'extrapillowsandblankets', 'family/kidfriendly', 'fireextinguisher',
       'firmmatress', 'firstaidkit', 'freeparkingonpremises',
       'gardenorbackyard', 'grab-railsforshowerandtoilet', 'heating',
       'indoorfireplace', 'iron', 'keypad', 'kitchen',
       'laptopfriendlyworkspace', 'lockbox', 'oven', 'pack’nplay/travelcrib',
       'patioorbalcony', 'pocketwifi', 'pool', 'privateentrance',
       'privatelivingroom', 'refrigerator',
       'roll-inshowerwithshowerbenchorchair', 'selfcheck-in',
       'singlelevelhome', 'smartlock', 'stairgates', 'stove',
       'translationmissing:en.hosting_amenity_49',
       'translationmissing:en.hosting_amenity_50', 'tv', 'widedoorway',
       'widehallwayclearance', 'property_last_updated', 'host_nr_listings',
       'booking_max_nights', 'property_beds', 'responsetime_code',
       'extra_beds', 'reviews_acc', 'reviews_cleanliness', 'reviews_checkin',
       'reviews_location', 'Host Is Superhost', 'Is Location Exact',
       'Require Guest Phone Verification', 'booking_availability_30',
       'booking_availability_60', 'reviews_num',
       'reviews_rating', 'booking_cancel_policy']]
#normalize features
ss=StandardScaler()
test_new=ss.fit_transform(test_do)

pred = model.predict(test_new)

print(pred)

ID=test[["property_id"]]

test['pred']=pred

prediction=test[['property_id', 'pred']]

prediction.to_csv('prediction.csv',index=False, sep=",")

