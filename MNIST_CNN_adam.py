import pandas as pd
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.datasets import mnist
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from keras.utils import to_categorical
import keras 

data_path = r'F:\StudyatCLass\Study\class\DeepLearningandUngdung\train.csv'
input_data = pd.read_csv(data_path)

r, c = 28, 28
num_classes = 10


(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], r, c, 1).astype('float32') / 255.0
X_test = X_test.reshape(X_test.shape[0], r, c, 1).astype('float32') / 255.0
input_shape = (r, c, 1)

 
y_train = to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model_path = r'F:\StudyatCLass\Study\class\DeepLearningandUngdung\MNIST_cnn_keras_adam.h5'


if os.path.exists(model_path):
    print(f"Đã tìm thấy model tại: {model_path}")
    model = load_model(model_path)
    model.summary()
else:
    print("Chưa có model, tiến hành tạo model mới và train...")
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    # model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adam(),
        metrics=['accuracy']
    )
    model.fit(
        X_train, y_train,
        batch_size=128,
        epochs=12,
        validation_data=(X_test, y_test),
        verbose=1
    )
    model.save(model_path)
    print(f"Đã lưu model vào file: {model_path}")

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

random_indices = np.random.randint(0, X_test.shape[0], 10)
x_random = X_test[random_indices]
y_random = y_test[random_indices]

y_pred = model.predict(x_random)

plt.figure(figsize=(12, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1 )
    plt.imshow(x_random[i].reshape(28 , 28), cmap='gray')
    plt.title(f"Dự đoán: {np.argmax(y_pred[i])}, Thực tế: {np.argmax(y_random[i])}")
    plt.axis('off')
plt.tight_layout()
plt.show()

y_test_pred = model.predict(X_test)
y_test_true_labels = np.argmax(y_test, axis=1)
y_test_pred_labels = np.argmax(y_test_pred, axis=1)


wrong_idx = np.where(y_test_true_labels != y_test_pred_labels)[0]
print("Số lượng ảnh đoán sai:", len(wrong_idx))

num_show = min(16, len(wrong_idx))

plt.figure(figsize=(12, 8))
for i in range(num_show):
    idx = wrong_idx[i]
    img = X_test[idx].reshape(28, 28)
    true_label = y_test_true_labels[idx]
    pred_label = y_test_pred_labels[idx]

    plt.subplot(4, 4, i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(f"Đoán: {pred_label}\nĐúng: {true_label}")
    plt.axis('off')

plt.tight_layout()
plt.show()

cm = confusion_matrix(y_test_true_labels, y_test_pred_labels)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Validation")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


# 4 bị đoán thành 9
indices_4_as_9 = np.where(
    (y_test_true_labels == 4) & (y_test_pred_labels == 9)
)[0]

# 9 bị đoán thành 4 (nếu cũng muốn xem luôn)
indices_9_as_4 = np.where(
    (y_test_true_labels == 9) & (y_test_pred_labels == 4)
)[0]

print("Số lần 4 bị đoán thành 9:", len(indices_4_as_9))
print("Số lần 9 bị đoán thành 4:", len(indices_9_as_4))

# Vẽ tất cả (hoặc tối đa 9) ảnh: true = 4, pred = 9
num_show = min(9, len(indices_4_as_9))

plt.figure(figsize=(12, 4))
for i in range(num_show):
    idx = indices_4_as_9[i]
    img = X_test[idx].reshape(28, 28)
    true_label = y_test_true_labels[idx]
    pred_label = y_test_pred_labels[idx]

    plt.subplot(2, 5, i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(f"Đoán: {pred_label}\nĐúng: {true_label}")
    plt.axis('off')

plt.tight_layout()
plt.show()
