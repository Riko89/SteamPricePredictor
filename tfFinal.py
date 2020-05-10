import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

URL = 'https://storage.googleapis.com/deeplearning-274123.appspot.com/Price_TagData.csv'
dataframe = pd.read_csv(URL)
dataframe.head()


train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('Price')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe),labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds. batch(batch_size)
  return ds 

batch_size = 5
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

for feature_batch, label_batch in train_ds.take(1):
  print('Every feature:', list(feature_batch.keys()))
  print('A batch of ages:', feature_batch['Action'])
  print('A batch of targets:', label_batch )

feature_columns = []

for header in ['Action', 'Adventure', 'Puzzle', 'RPG', 'Strategy']:
  feature_columns.append(feature_column.numeric_column(header))
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(128, activation='relu'),
  layers.Dense(128, activation='relu'),
  layers.Dense(1)                           
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(train_ds,
          validation_data=val_ds,
          epochs=10)
loss, accuracy = model.evaluate(test_ds)
#prediction = model.predict(pd.DataFrame({'Action':[0], 'Adventure':[0], 'Puzzle':[0], 'RPG':[0],'Strategy':[0]}))
#print("Accuracy: ", accuracy)