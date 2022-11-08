from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

df = pd.read_csv('C:\data\house_train.csv')

df = pd.get_dummies(df)

df = df.fillna(df.mean())

df_corr = df.corr()

df_corr_sort = df_corr.sort_values('SalePrice', ascending=False)

cols_train = ['OverallQual', 'GrLivArea',
              'GarageCars', 'GarageArea', 'TotalBsmtSF']
x_train_pre = df[cols_train]

y = df['SalePrice'].values

x_train, x_test, y_train, y_test = train_test_split(
    x_train_pre, y, test_size=0.2)

model = Sequential()
model.add(Dense(10, input_dim=x_train.shape[1], activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(1))
model.summary()

model.compile(optimizer='adam', loss='mean_squared_error')

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=20)

modelpath = "./data/model/Ch15-house.hdf5"

checkpointer = ModelCheckpoint(
    filepath=modelpath, monitor='val_loss', verbose=0, save_best_only=True)

history = model.fit(x_train, y_train, validation_batch_size=0.25, epochs=2000,
                    batch_size=32, callbacks=[early_stopping_callback, checkpointer])

real_prices = []
pred_prices = []
x_num = []

n_iter = 0
y_prediction = model.predict(x_test).flatten()
for i in range(25):
    real = y_test[i]
    prediction = y_prediction[i]
    print('실제 가격:{:.2f}, 예상 가격: {:.2f}'.format(real, prediction))
    real_prices.append(real)
    pred_prices.append(prediction)
    n_iter = n_iter + 1
    x_num.append(n_iter)

plt.plot(x_num, pred_prices, label='predicted price')
plt.plot(x_num, real_prices, label='real price')
plt.legend()
plt.show()
