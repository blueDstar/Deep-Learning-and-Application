import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ======= 1. Load dữ liệu =======
data_path = r'F:\StudyatCLass\Study\class\DeepLearningandUngdung\train.csv'
input_data = pd.read_csv(data_path)

y = input_data['label']
X = input_data.drop('label', axis=1) / 255.0  # Chuẩn hóa pixel

y = pd.get_dummies(y)

X_train, X_vali, y_train, y_vali = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======= 2. Đường dẫn lưu mô hình =======
model_path = r'F:\StudyatCLass\Study\class\DeepLearningandUngdung\ann_model2.h5'

# ======= 3. Kiểm tra nếu model đã lưu thì load, nếu chưa train mới =======
if os.path.exists(model_path):
    print("Đang load model đã lưu...")
    classifier2 = load_model(model_path)
else:
    print("Đang train model mới...")
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

    # Lưu model sau khi train xong
    classifier2.save(model_path)
    print(f"Đã lưu model tại: {model_path}")

# ======= 4. Dự đoán và tính accuracy =======
y_pred_vali = classifier2.predict(X_vali)

accuracy = accuracy_score(
    y_vali.values.argmax(axis=1),
    y_pred_vali.argmax(axis=1)
)
print(f'Độ chính xác trên tập validation: {accuracy*100:.2f}%')

# ======= 5. Vẽ đồ thị accuracy và loss nếu vừa train =======
if 'history' in locals():
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Accuracy Train vs Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss Train vs Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

# ======= 6. Hiển thị 10 ảnh ngẫu nhiên cùng dự đoán =======
num_samples = 10
random_indices = np.random.choice(len(X_vali), num_samples, replace=False)
sample_images = X_vali.iloc[random_indices]
sample_labels = y_vali.iloc[random_indices]

pred_probs = classifier2.predict(sample_images)
sample_predictions = np.argmax(pred_probs, axis=1)
sample_true = np.argmax(sample_labels.values, axis=1)

plt.figure(figsize=(15, 5))
for i in range(num_samples):
    plt.subplot(2, 5, i + 1)
    plt.imshow(sample_images.iloc[i].values.reshape(28, 28), cmap='gray')
    plt.title(f"Thật: {sample_true[i]}\nDự đoán: {sample_predictions[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
