import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

input_data = pd.read_csv(r'F:\StudyatCLass\Study\class\DeepLearningandUngdung\train.csv')

y_train = input_data['label']
input_data.drop('label', axis = 1, inplace = True)
X_train = input_data
y_train = pd.get_dummies(y_train)

test_data = pd.read_csv(r'F:\StudyatCLass\Study\class\DeepLearningandUngdung\test.csv')
X_test = test_data

classifier = Sequential()
classifier.add(Dense(600, activation='relu', input_shape=(784,) , use_bias = True))
classifier.add(Dense(400, activation='relu', use_bias = True))
classifier.add(Dense(200, activation='relu', use_bias = True))
classifier.add(Dense(10, activation='sigmoid', use_bias = True))

classifier.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])

classifier.fit(X_train, y_train, epochs=20, batch_size=30)

y_pred = np.argmax(classifier.predict(X_test), axis=1)

num_samples = 10
random_indices = np.random.choice(X_test.shape[0], num_samples, replace = False)
sample_images = X_test.iloc[random_indices]
sample_predictions = y_pred[random_indices]

plt.figure(figsize=(15, 5))
for i, idx in enumerate(random_indices):
    plt.subplot(2, 5, i + 1 )
    plt.imshow(sample_images.iloc[i].values.reshape(28 , 28), cmap='gray')
    plt.title(f"Dự đoán: {sample_predictions[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()