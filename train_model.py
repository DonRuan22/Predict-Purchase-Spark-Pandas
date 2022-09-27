# -*- coding: utf-8 -*-
#import machine learning related libraries
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

from google.cloud import bigquery
import gcsfs
import joblib
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("airflow.task")


bigqueryClient = bigquery.Client()
query_string = """
SELECT
*
FROM `donexp.onlineRetail.onlineRetailTransformed`
"""
tx_class = (
    bigqueryClient.query(query_string)
    .result()
    .to_dataframe(
        # Optionally, explicitly request to use the BigQuery Storage API. As of
        # google-cloud-bigquery version 1.26.0 and above, the BigQuery Storage
        # API is used by default.
        create_bqstorage_client=True,
    )
)
    

tx_classdb = pd.get_dummies(tx_class)

tx_classdb['NextPurchaseDayRange'] = 2
tx_classdb.loc[tx_classdb.NextPurchaseDay>20,'NextPurchaseDayRange'] = 1
tx_classdb.loc[tx_classdb.NextPurchaseDay>50,'NextPurchaseDayRange'] = 0
    
logger.info('dataframe head - {}'.format(tx_classdb.describe()))  
logger.info('dataframe head - {}'.format(tx_classdb.NextPurchaseDayRange))  
logger.info('dataframe head - {}'.format(tx_classdb.NextPurchaseDay))  


#train & test split
tx_classdb = tx_classdb.drop('NextPurchaseDay',axis=1)
X, y = tx_classdb.drop('NextPurchaseDayRange',axis=1), tx_classdb.NextPurchaseDayRange
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)

kfold = KFold(n_splits=2,shuffle=True, random_state=22)
cv_result = cross_val_score(xgb.XGBClassifier(),X_train,y_train, cv = kfold,scoring = "accuracy")
#print('Xboost '+ cv_result)
logger.info('Xboost - {}'.format(cv_result))
xgb_model = xgb.XGBClassifier(max_depth=5,min_child_weight=5).fit(X_train,y_train)
logger.info("Accuracy of XBClassifier on training dataset:{:.2f}"
      .format(xgb_model.score(X_train,y_train)))
logger.info("Accuracy of XBClassifier on test dataset:{:.2f}"
      .format(xgb_model.score(X_test,y_test)))

filename = 'gcs://don-onlineretail/predict_purchase_model.joblib.pkl'
fs = gcsfs.GCSFileSystem()
with fs.open(filename, 'wb') as f:
    joblib.dump(xgb_model, f, compress=9)
    #xgb_model.save_model(filename)
#with fs.open(filename, 'rb') as f:
    #model = joblib.load(f)
    #cols_when_model_builds = model.get_booster().feature_names
    #logger.info('Columns - {}'.format(cols_when_model_builds))
    #X_test = X_test[cols_when_model_builds]
    #predicted = model.predict(X_test)
    #logger.info('Predicted - {}'.format(predicted))
