# train_model.py

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from mtcnn.mtcnn import MTCNN
import cv2

# Ensure reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Define constants
IMAGE_SIZE = (300, 300) 
BATCH_SIZE = 32
EPOCHS = 50 # Increase epochs since we are training from scratch
LEARNING_RATE = 1e-4
DATA_DIR = r"C:\Users\BAPS\Downloads\gender_13k\Train\Train" # !!! CHANGE THIS PATH !!!
MODEL_SAVE_PATH = 'gender_detection_efficientnetb3.keras'

# Step 1: Data Preparation using ImageDataGenerator
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2 
)

train_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary', 
    subset='training',
    color_mode='rgb'
)

validation_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    color_mode='rgb'
)

# Step 2: Build the EfficientNetB3 model
# Set `weights=None` to initialize the model with random weights and train from scratch.
base_model = EfficientNetB3(
    weights=None,
    include_top=False,
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
)

# Since we are training from scratch, all layers should be trainable.
# We will not freeze the base model layers.

# Add custom layers for gender detection
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x) 

model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Step 3: Train the model
# Use callbacks to save the best model and stop training early if accuracy plateaus.
checkpoint = ModelCheckpoint(
    MODEL_SAVE_PATH,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10, # Increase patience for training from scratch
    restore_best_weights=True
)

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[checkpoint, early_stopping]
)

print("Model training complete.")