import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
# Tải tập huấn luyện
train_data = pd.read_csv(r"F:\StudyatCLass\Study\class\DeepLearningandUngdung\train.csv")
# Lấy nhãn
y_train = train_data['label']
# Bỏ nhãn khỏi tập pixel
train_data.drop('label', axis=1, inplace=True)  
X_train = train_data
X_train, X_vali, y_train, y_vali = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)
# One-hot encode cho nhãn
y_train = pd.get_dummies(y_train)  

from keras.models import Sequential
from keras.layers import Dense


classifier = Sequential()
classifier.add(Dense(units=600, kernel_initializer='uniform',
                     activation='relu', input_dim=784))
classifier.add(Dense(units=400, kernel_initializer='uniform',
                     activation='relu'))
classifier.add(Dense(units=200, kernel_initializer='uniform',
                     activation='relu'))
classifier.add(Dense(units=10, kernel_initializer='uniform',
                     activation='sigmoid'))

classifier.compile(optimizer='sgd', loss='mean_squared_error',
                   metrics=['accuracy'])

history = classifier.fit(X_train, y_train, batch_size=10, epochs=10)

# Tải tập kiểm tra
test_data = pd.read_csv(r"F:\StudyatCLass\Study\class\DeepLearningandUngdung\test.csv")
X_test = test_data

import matplotlib.pyplot as plt
import numpy as np

# Dự đoán nhãn cho tập kiểm tra X_test
y_pred = np.argmax(classifier.predict(X_test), axis=1)


# Chọn 10 mẫu ngẫu nhiên từ tập kiểm tra
num_samples = 10
random_indices = np.random.choice(X_test.shape[0], num_samples, replace=False)
sample_images = X_test.iloc[random_indices]
sample_predictions = y_pred[random_indices]

# Hiển thị hình ảnh và nhãn dự đoán
plt.figure(figsize=(15, 5))
for i, idx in enumerate(random_indices):
    plt.subplot(2, 5, i + 1)
    plt.imshow(sample_images.iloc[i].values.reshape(28, 28), cmap='gray')
    plt.title(f"Pred: {sample_predictions[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

accuracy = accuracy_score(y_vali.values.argmax(axis=1),
                               y_pred.argmax(axis=1))
print(f"Accuracy: {accuracy:.2f}")