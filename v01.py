from __future__ import print_function
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.cross_validation import train_test_split
import numpy as np
import xgboost as xgb
print(pd.__version__)
print("Loading data...")
df = pd.read_csv('train.csv', index_col='QuoteNumber', thousands=',', parse_dates=True)
# clean data
print("Cleaning data...")
df_clean = df.copy()
df_clean['Year'] = pd.to_datetime(df_clean['Original_Quote_Date']).map(lambda x: x.year)
df_clean['Month'] = pd.to_datetime(df_clean['Original_Quote_Date']).map(lambda x: x.month)
df_clean['Month'] = pd.to_datetime(df_clean['Original_Quote_Date']).map(lambda x: x.day)
# remove Original_Quote_Date
df_clean.drop('Original_Quote_Date', axis=1, inplace=True)
df_clean.drop('PropertyField6', axis=1, inplace=True)
df_clean.drop('GeographicField10A', axis=1, inplace=True)
mapping = {}
for column in df.loc[:, df.dtypes == object]:
    mapping[column] = dict(zip(df[column].unique(),np.arange(len(df[column].unique()))))

df_clean.replace(mapping, inplace=True)
y = df_clean.fillna(-1).iloc[:, 0:1].values.reshape((df_clean.shape[0]))
X = df_clean.fillna(-1).iloc[:, 1:].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# model = LogisticRegressionCV(n_jobs=-1)
model = xgb.XGBClassifier(n_estimators=25, nthread=-1, max_depth=10, learning_rate=0.025, silent=True, subsample=0.8, colsample_bytree=0.8)
print("Training model...")
model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], eval_metric="auc")
mddel.evals_result()
