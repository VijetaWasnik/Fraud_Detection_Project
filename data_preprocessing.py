import numpy as np 
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.metrics import recall_score, f1_score, make_scorer, cohen_kappa_score
from sklearn.preprocessing import MinMaxScaler
from category_encoders import OneHotEncoder

data = pd.read_csv("PS_20174392719_1491204439457_log.csv")

# difference between initial balance before the transaction and new balance after the transaction
data['diff_new_old_balance'] = data['newbalanceOrig'] - data['oldbalanceOrg']

# difference between initial balance recipient before the transaction and new balance recipient after the transaction.
data['diff_new_old_destiny'] = data['newbalanceDest'] - data['oldbalanceDest']

# name orig and name dest
data['nameOrig'] = data['nameOrig'].apply(lambda i: i[0])
data['nameDest'] = data['nameDest'].apply(lambda i: i[0])

X = data.drop(columns=['isFraud', 'isFlaggedFraud', 'nameOrig', 'nameDest'], axis = 1)

y = data['isFraud']

# spliting into temp and test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=.2, stratify=y)

# spliting into train and valid
X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=.2, stratify=y_temp)

ohe = OneHotEncoder(cols=['type'], use_cat_names=True)

X_train = ohe.fit_transform(X_train)
X_valid = ohe.transform(X_valid)

X_temp = ohe.fit_transform(X_temp)
X_test = ohe.transform(X_test)

num_columns = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
               'diff_new_old_balance', 'diff_new_old_destiny']
scaler = MinMaxScaler()

X_params = X_temp.copy()

X_train[num_columns] = scaler.fit_transform(X_train[num_columns])
X_valid[num_columns] = scaler.transform(X_valid[num_columns])

X_params[num_columns] = scaler.fit_transform(X_temp[num_columns])
X_test[num_columns] = scaler.transform(X_test[num_columns])

final_columns_selected = ['step', 'oldbalanceOrg', 
                          'newbalanceOrig', 'newbalanceDest', 
                          'diff_new_old_balance', 'diff_new_old_destiny', 
                          'type_TRANSFER']

# Select features for training, validation, temporary data, test data, and parameters data
X_train_cs = data.loc[:, final_columns_selected]
X_valid_cs = data.loc[:, final_columns_selected]

X_temp, X_test = train_test_split(data, test_size=0.5, stratify=data['isFraud'])
X_temp_cs = X_temp.loc[:, final_columns_selected]
X_test_cs = X_test.loc[:, final_columns_selected]

X_params_cs = data.loc[:, final_columns_selected]

X_train_cs.to_csv('X_train.csv', index=False)
X_valid_cs.to_csv('X_valid.csv', index=False)
X_test_cs.to_csv('X_test.csv', index=False)
X_temp_cs.to_csv('X_temp.csv', index=False)
X_params_cs.to_csv('X_params.csv', index=False)
y_temp.to_csv('y_temp.csv', index=False)
y_valid = pd.read_csv('y_valid.csv')


with open('ohe_encoder.pkl', 'wb') as f:
    pickle.dump(ohe, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)