import numpy as np 
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

X = np.array([[0.5,0.3]])
y = np.array([[0]])

model = Sequential()
model.add(Dense(2, input_dim =2, activation = 'relu', use_bias = True))
model.add(Dense(1, activation = 'sigmoid', use_bias = True))

model.compile(loss ='binary_crossentropy', optimizer = 'sgd', metrics = ['accuracy'])

model.layers[0].set_weights([np.array([[0.4, 0.2],[0.6, 0.7]]),
                             np.array([0.1, 0.1])])
model.layers[1].set_weights([np.array([[0.5], [0.3]]), np.array([0.2])])

prediction = model.predict(X)
print(f'Dự đoán đầu ra của mạng nơ-ron: {prediction[0][0]}')