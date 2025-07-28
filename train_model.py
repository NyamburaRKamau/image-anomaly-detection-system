import numpy as np

# Load Dataset
data = np.load('/Users/RoseKamau/Downloads/image_anomaly_data.npz')
images =data['images']
labels = data['labels']

print('Shape of images:', images.shape)
print('Shape of labels:', labels.shape)
print('Unique labels:', np.unique(labels))


import matplotlib.pyplot as plt

# Function to plot images
def plot_images(images, labels, title, count=5):
    plt.figure(figsize=(15, 3))
    for i in range(count):
        plt.subplot(1, count, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(f'{title} #{i+1}')
        plt.axis('off')
    plt.show()

# Separate normal and anomalous
normal_images = images[labels == 0]
anomaly_images = images[labels == 1]

# Visualize
print("Normal Images:")
plot_images(normal_images, labels[labels == 0], title='Normal')

print('Anomalous Images:')
plot_images(anomaly_images, labels[labels == 1], title='Anomaly')


from sklearn.model_selection import train_test_split

# Reshape for CNN (add chanel dimension)
images_reshaped = images.reshape(-1, 28, 28, 1)

# Normalize (pixel values between 0 and 1)
images_normalized = images_reshaped / 255.0

# Train-Test-Split
X_train, X_test, y_train, y_test = train_test_split(
    images_normalized, labels, test_size=0.2, random_state=42, stratify=labels
)
print('Training set shape:', X_train.shape)
print('Testing set shape:', X_test.shape)

import tensorflow as tf
from tensorflow.keras import layers, models

# Build the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32,
                    validation_data=(X_test, y_test))


import matplotlib.pyplot as plt

# Plot training & validation accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Predict on test set
y_pred = model.predict(X_test)
y_pred_labels = (y_pred > 0.5).astype(int).flatten()

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_labels)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Classification report
print(classification_report(y_test, y_pred_labels, target_names=['Normal','Anomaly']))


import random

# Get predictions
preds = model.predict(X_test)
predicted_labels = (preds > 0.5).astype("int32").flatten()

# Visualize 10 random test images with predictions
plt.figure(figsize=(15, 6))
for i in range(10):
    idx = random.randint(0, len(X_test) -1)
    image = X_test[idx].reshape(28, 28)
    true_label = y_test[idx]
    pred_label = predicted_labels[idx]


    plt.subplot(2, 5, i + 1)
    plt.imshow(image, cmap='gray')
    plt.title(f'True; {true_label} | pred: {pred_label}')
    plt.axis('off')

plt.tight_layout()
plt.show()

import joblib

# Save model and scaler
model.save('anomaly_model.h5')
