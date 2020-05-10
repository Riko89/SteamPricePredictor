from io import StringIO, BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib


url =  'https://storage.googleapis.com/deeplearning-274123.appspot.com/Price_TagData.csv'
dataset = pd.read_csv(url)
dataset.head(15)

feature_cols = ['Action', 'Adventure', 'Puzzle', 'RPG', 'Strategy']
X = dataset[feature_cols]
y = dataset.Price

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

print("Accuracy:",regressor.score(X_train, y_train))

Price =regressor.predict(pd.DataFrame({'Action':[1], 'Adventure':[1], 'Puzzle':[0], 'RPG':[0],'Strategy':[0]}))
print("Price prediction: ", Price)

filename = 'finalized_model.pkl'
joblib.dump(regressor, filename)