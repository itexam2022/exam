import numpy as np
df = np.loadtxt("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",delimiter=',')
df
#remember here df is not data frame it is an array

x = df[:,:8]
y = df[:,8]

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()

#input layer
model.add(Dense(12, input_dim=8, activation='relu'))

#hidden layer
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))

#output layer // for binary classification - sigmoid, multicalss - softmax
model.add(Dense(1, activation="sigmoid"))

#2Compile the model
model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracyz'])

#train model
model.fit(x, y, epochs=100, batch_size=10)

#3evalution
model.evaluate(x,y)