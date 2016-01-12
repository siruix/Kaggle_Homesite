
from __future__ import print_function
import pandas as pd
# from sklearn.linear_model import LogisticRegressionCV
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from datetime import datetime

print("Loading data...")
df = pd.read_csv('train.csv', index_col='QuoteNumber', thousands=',', parse_dates=True)
# clean data
print("Cleaning...")
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

y = df_clean.fillna(-1).iloc[:, 0].values.reshape((df_clean.shape[0]))
X = df_clean.fillna(-1).iloc[:, 1:].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

params = {'n_estimators': 10, 'max_depth': 3, 'min_samples_split': 1, 'learning_rate': 1, 'verbose':1}
model = ensemble.GradientBoostingClassifier(**params)
print("Training...")
start_time = datetime.now()
model.fit(X_train, y_train)
end_time = datetime.now()
print('Training Duration: {}'.format(end_time - start_time))

mse_test = mean_squared_error(y_test, model.predict(X_test))
mse_train = mean_squared_error(y_train, model.predict(X_train))
print("MSE_train: %.4f" % mse_train)
print("MSE_test: %.4f" % mse_test)

print("Train score: %.4f" % model.score(X_train, y_train))
print("Test score: %.4f" % model.score(X_test, y_test))