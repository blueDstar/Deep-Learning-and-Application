import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ======= 1. Load dữ liệu train =======
data_path = r'F:\StudyatCLass\Study\class\DeepLearningandUngdung\train.csv'
input_data = pd.read_csv(data_path)

y = input_data['label']
X = input_data.drop('label', axis=1) / 255.0
y = pd.get_dummies(y)

X_train, X_vali, y_train, y_vali = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======= 2. Đường dẫn lưu mô hình =======
model_path = r'F:\StudyatCLass\Study\class\DeepLearningandUngdung\ann_model2.h5'

# ======= 3. Load hoặc train model =======
if os.path.exists(model_path):
    print("Loading model đã lưu...")
    classifier2 = load_model(model_path)
else:
    print("Training model mới...")
    classifier2 = Sequential()
    classifier2.add(Dense(512, activation='relu', input_shape=(784,)))
    classifier2.add(BatchNormalization())
    classifier2.add(Dropout(0.3))
    classifier2.add(Dense(256, activation='relu'))
    classifier2.add(Dropout(0.3))
    classifier2.add(Dense(128, activation='relu'))
    classifier2.add(Dense(10, activation='softmax'))

    classifier2.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history = classifier2.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_data=(X_vali, y_vali),
        verbose=1
    )

    classifier2.save(model_path)
    print(f"Model đã lưu tại: {model_path}")

# ======= 4. Dự đoán trên validation =======
y_pred_vali = classifier2.predict(X_vali)
y_vali_labels = y_vali.values.argmax(axis=1)
y_pred_vali_labels = y_pred_vali.argmax(axis=1)

accuracy = accuracy_score(y_vali_labels, y_pred_vali_labels)
print(f'Độ chính xác trên tập validation: {accuracy*100:.2f}%')

# ======= 5. Confusion matrix =======
cm = confusion_matrix(y_vali_labels, y_pred_vali_labels)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Validation")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# ======= 6. Hiển thị 10 ảnh validation ngẫu nhiên =======
num_samples = 10
random_indices = np.random.choice(len(X_vali), num_samples, replace=False)
sample_images = X_vali.iloc[random_indices]
sample_labels = y_vali_labels[random_indices]
sample_predictions = y_pred_vali_labels[random_indices]

plt.figure(figsize=(15, 5))
for i in range(num_samples):
    plt.subplot(2, 5, i + 1)
    plt.imshow(sample_images.iloc[i].values.reshape(28, 28), cmap='gray')
    plt.title(f"Thật: {sample_labels[i]}\nDự đoán: {sample_predictions[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# ======= 7. Dự đoán trên test.csv (không có nhãn) =======
test_path = r'F:\StudyatCLass\Study\class\DeepLearningandUngdung\test.csv'
X_test = pd.read_csv(test_path) / 255.0

y_pred_test_probs = classifier2.predict(X_test)
y_pred_test = np.argmax(y_pred_test_probs, axis=1)

# ======= 8. Hiển thị 10 ảnh test ngẫu nhiên với dự đoán =======
num_samples_test = 10
random_indices_test = np.random.choice(len(X_test), num_samples_test, replace=False)
sample_test_images = X_test.iloc[random_indices_test]
sample_test_predictions = y_pred_test[random_indices_test]

plt.figure(figsize=(15, 5))
for i in range(num_samples_test):
    plt.subplot(2, 5, i + 1)
    plt.imshow(sample_test_images.iloc[i].values.reshape(28, 28), cmap='gray')
    plt.title(f"Dự đoán: {sample_test_predictions[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
