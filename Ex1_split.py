import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
input_data = pd.read_csv(r'F:\StudyatCLass\Study\class\DeepLearningandUngdung\train.csv')

y = input_data['label']
X = input_data.drop('label', axis=1)
X = X / 255.0
y = pd.get_dummies(y)

X_train, X_vali, y_train, y_vali = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_path = r'F:\StudyatCLass\Study\class\DeepLearningandUngdung\ann_model1.h5'

if os.path.exists(model_path):
    print("Loading model đã lưu...")
    classifier = load_model(model_path)
else:
    print("Training model mới...")

    classifier = Sequential()
    classifier.add(Dense(600, activation='relu', input_shape=(784,)))
    classifier.add(Dense(400, activation='relu'))
    classifier.add(Dense(200, activation='relu'))
    classifier.add(Dense(10, activation='sigmoid'))

    classifier.compile(optimizer='sgd', 
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    classifier.fit(X_train, y_train, epochs=20, batch_size=30)

    classifier.save(model_path)
    print(f"Model đã lưu tại: {model_path}")

y_pred_vali = classifier.predict(X_vali)

accuracy = accuracy_score(
    y_vali.values.argmax(axis=1),
    y_pred_vali.argmax(axis=1)
)




test_data = pd.read_csv(r'F:\StudyatCLass\Study\class\DeepLearningandUngdung\test.csv')
X_test = test_data / 255.0   # chuẩn hóa

print(f'Độ chính xác trên tập validation: {accuracy*100:.2f}%')

y_pred_test = np.argmax(classifier.predict(X_test), axis=1)

num_samples = 10
random_indices = np.random.choice(X_test.shape[0], num_samples, replace=False)
sample_images = X_test.iloc[random_indices]
sample_predictions = y_pred_test[random_indices]

cm = confusion_matrix(y_vali, y_pred_vali)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Validation")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

plt.figure(figsize=(15, 5))
for i in range(num_samples):
    plt.subplot(2, 5, i + 1)
    plt.imshow(sample_images.iloc[i].values.reshape(28, 28), cmap='gray')
    plt.title(f"Dự đoán: {sample_predictions[i]}")
    plt.axis('off')

plt.tight_layout()
plt.show()
