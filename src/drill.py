import tensorflow as tf
from keras.datasets.mnist import load_data
from keras.models import Sequential
from keras import models
from keras.layers import Dense, Input, Flatten, Dropout
from keras.utils import to_categorical, plot_model
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
from keras.optimizers import Adam
import numpy as np
import pandas as pd
import processer as util
from sklearn.model_selection import train_test_split

  
model = Sequential([
            Input(shape=(14)),
            Dense(18, activation='relu'),
            Dense(9, activation='relu'),
            Dense(3, activation="sigmoid")
        ])

x_data = []
y_data = []

loader = util.CSVLoader()
# processer = util.MultiProcesser(
#     [
#         util.AngleProcesser(),
#         util.DistanceProcesser2(),
#     ]
# )
processer = util.AngleProcesser()

stand_data = loader('./data/stand/dataset.csv')
left_data = loader('./data/left/dataset.csv')
right_data = loader('./data/right/dataset.csv')

for load in stand_data:
    x_data.append(processer(load))
    y_data.append(1)
for load in left_data:
    x_data.append(processer(load))
    y_data.append(0)
for load in right_data:
    x_data.append(processer(load))
    y_data.append(2)

x_data = np.array(x_data)
y_data = np.array(y_data)

x_train_data, x_val_data, y_train_data, y_val_data = train_test_split(x_data, y_data, test_size=0.1, random_state=1, shuffle=False)

model.compile(optimizer=SGD(learning_rate=0.01),
          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
          metrics=['accuracy'])

model.fit(
    x=x_train_data,
    y=y_train_data,
    validation_data=(x_val_data, y_val_data),
    epochs=32, 
    batch_size=32, 
    shuffle=True
)

model.save('./dist/temp.h5')