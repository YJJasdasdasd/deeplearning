from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import pandas as pd

df = pd.read_csv('C:\data\wine.csv', header=None)

df

x = df.iloc[:, 0:12]
y = df.iloc[:, 12]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, shuffle=True)

model = Sequential()
model.add(Dense(30, input_dim=12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=20)


history = model.fit(x_train, y_train, epochs=50, batch_size=200)
score = model.evaluate(x_test, y_test)
print('Test accuracy:', score[1])
