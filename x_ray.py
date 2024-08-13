#!/usr/bin/env python
# coding: utf-8

# In[9]:


import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# # Load Image Dataset

# In[2]:


# Define the path to the dataset
base_path = r"C:\Users\rakat.murshed\Documents\Datasets\x_ray\chest_xray\train"

# Define the subfolders
subfolders = ['NORMAL', 'PNEUMONIA']

# Collect all image paths and labels
image_paths = []
labels = []

for subfolder in subfolders:
    folder_path = os.path.join(base_path, subfolder)
    if os.path.exists(folder_path):
        for file in os.listdir(folder_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(folder_path, file))
                labels.append(subfolder)

print(f"Total number of images: {len(image_paths)}")


# # Image Processing

# In[3]:


# Define image size and batch size
image_size = (224, 224)
batch_size = 32

# Function to preprocess images
def preprocess_image(image_path, target_size=image_size):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Normalize the image
    return img


# In[4]:


# Preprocess all images
X = np.array([preprocess_image(img_path) for img_path in image_paths])
y = np.array([1 if label == 'PNEUMONIA' else 0 for label in labels])


# In[5]:


# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


# In[6]:


# Show 5 preprocessed images
plt.figure(figsize=(15, 5))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(X[i], cmap='gray')
    plt.title('PNEUMONIA' if y[i] == 1 else 'NORMAL')
    plt.axis('off')
plt.show()


# # Build CNN Model

# In[7]:


# Define the custom CNN model
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(*image_size, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model


# In[10]:


kf = KFold(n_splits=3, shuffle=True, random_state=42)  # Reduced splits
epochs = 10  # Reduced epochs

train_mse = []
train_r2 = []
val_mse = []
val_r2 = []

for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    model = create_model()
    
    history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
                        validation_data=(X_val, y_val),
                        epochs=epochs,
                        callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)])
    
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    train_mse.append(mean_squared_error(y_train, y_train_pred))
    train_r2.append(r2_score(y_train, y_train_pred))
    val_mse.append(mean_squared_error(y_val, y_val_pred))
    val_r2.append(r2_score(y_val, y_val_pred))
    
    print(f"Train MSE: {train_mse[-1]}, Train R2: {train_r2[-1]}")
    print(f"Val MSE: {val_mse[-1]}, Val R2: {val_r2[-1]}")


# # Error Analysis

# In[13]:


fold_range = range(1, len(train_mse) + 1)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.bar(fold_range, train_mse, label='Train MSE')
plt.xlabel('Fold')
plt.ylabel('Mean Squared Error')
plt.title('Training Mean Squared Error')
plt.legend()

plt.subplot(1, 2, 2)
plt.bar(fold_range, val_mse, label='Validation MSE')
plt.xlabel('Fold')
plt.ylabel('Mean Squared Error')
plt.title('Validation Mean Squared Error')
plt.legend()

plt.show()

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.bar(fold_range, train_r2, label='Train R2 Score')
plt.xlabel('Fold')
plt.ylabel('R2 Score')
plt.title('Training R2 Score')
plt.legend()

plt.subplot(1, 2, 2)
plt.bar(fold_range, val_r2, label='Validation R2 Score')
plt.xlabel('Fold')
plt.ylabel('R2 Score')
plt.title('Validation R2 Score')
plt.legend()

plt.show()


# In[ ]:




