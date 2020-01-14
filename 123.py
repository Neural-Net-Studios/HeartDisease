from keras.models import Sequential
from keras import layers
import pandas as pd
from sklearn.model_selection import train_test_split
import os
data = pd.read_csv('C:/Users/asynk/Desktop/Data/Beauty.csv')
x = data.drop(['looks'], axis=1)
y = data['looks']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35)
# Model
model = Sequential()
# Input Layer
model.add(layers.Dense(300, activation='relu', input_dim=9))
# Hidden Layer 1
model.add(layers.Dropout(0.35))
model.add(layers.Dense(200, activation='relu'))
# Output Layer
model.add(layers.Dense(1, activation='relu'))
# Summary
model.summary()
# Compilation
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['accuracy']
)
if not os.path.isfile('model_1'):
    # Training
    model.fit(x_train, y_train, epochs=110)
    model.save_weights('model_1')
else:
    model.load_weights('model_1')
# Accuracy
score = model.evaluate(x_test, y_test)
print(score[1])
