from __future__ import print_function
import pandas as pd
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn import ensemble
import logging
from datetime import datetime
from sklearn.externals import joblib

logging.basicConfig(filename='gb.log', level=logging.DEBUG)
logging.info(str(datetime.now()))
logging.info("Loading data ...")
df = pd.read_csv('train.csv', index_col='QuoteNumber', thousands=',', parse_dates=True)
# clean data
df_clean = df.copy()
df_clean['Year'] = pd.to_datetime(df_clean['Original_Quote_Date']).map(lambda x: x.year)
df_clean['Month'] = pd.to_datetime(df_clean['Original_Quote_Date']).map(lambda x: x.month)
df_clean['Month'] = pd.to_datetime(df_clean['Original_Quote_Date']).map(lambda x: x.day)
# remove Original_Quote_Date
df_clean.drop('Original_Quote_Date', axis=1, inplace=True)
df_clean.drop('PropertyField6', axis=1, inplace=True)
df_clean.drop('GeographicField10A', axis=1, inplace=True)
mapping = {}
for column in df_clean.loc[:, df_clean.dtypes == object]:
    mapping[column] = dict(zip(df_clean[column].unique(),np.arange(len(df_clean[column].unique()))))

df_clean.replace(mapping, inplace=True)
y = df_clean.fillna(-999).iloc[:, 0].values.reshape((df_clean.shape[0]))
X = df_clean.fillna(-999).iloc[:, 1:].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# training with varies parameters
original_params = {'n_estimators': 1, 'max_depth': 5, 'min_samples_split': 1, 'subsample':1, 'learning_rate': 0.1, 'verbose':1}
train_deviance = np.zeros((6 ,original_params['n_estimators']), dtype=np.float64)
test_deviance = np.zeros((6, original_params['n_estimators']), dtype=np.float64)
for i, (label, color, setting) in enumerate( [('No shrinkage', 'orange',
                               {'learning_rate': 1.0, 'subsample': 1.0}),
                              ('learning_rate=0.1', 'turquoise',
                               {'learning_rate': 0.1, 'subsample': 1.0}),
                              ('subsample=0.8', 'blue',
                               {'learning_rate': 1.0, 'subsample': 0.8}),
                              ('learning_rate=0.1, subsample=1.0', 'yellow',
                               {'learning_rate': 0.1, 'max_features': 0.9}),
                              ('learning_rate=0.1, max_features=2', 'magenta',
                               {'learning_rate': 0.1, 'max_features': 2}),
                              ('learning_rate=0.1, subsample=0.8', 'gray',
                               {'learning_rate': 0.1, 'subsample': 0.8}),] ):
    params = dict(original_params)
    params.update(setting)
    model = ensemble.GradientBoostingClassifier(**params)
    logging.info("Training with parameters: {0} ...".format(label))
    logging.info(params)
    model.fit(X_train, y_train)

    
    train_deviance[i] = model.train_score_
    for j, y_pred in enumerate(model.staged_decision_function(X_test)):
        # clf.loss_ assumes that y_test[i] in {0, 1}
        test_deviance[i][j] = model.loss_(y_test, y_pred)
	joblib.dump(model, 'model_{0}.pkl'.format(label))     
	# save data after each run in case lose connection
	train_deviance.tofile('train_deviance_{0}.dat'.format(label))
	test_deviance.tofile('test_deviance_{0}.dat'.format(label))