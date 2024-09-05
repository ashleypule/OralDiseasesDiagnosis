import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import VGG16
import json
import os

# Configuration
img_height, img_width = 150, 150
batch_size = 32
dataset_path = 'Dataset'  # Ensure this path is correct

# Check if the dataset path exists
if not os.path.isdir(dataset_path):
    raise ValueError(f"The dataset path {dataset_path} does not exist or is not a directory.")

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Adding validation split
)

train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Save class labels to JSON
class_labels = {v: k for k, v in train_generator.class_indices.items()}
with open('class_labels.json', 'w') as f:
    json.dump(class_labels, f)

# Load pre-trained VGG16 model and add custom layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)  # Adding Dropout for regularization
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Save model checkpoint
checkpoint = ModelCheckpoint('oral_disease_model.keras', monitor='val_loss', save_best_only=True)

# Train model
model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    callbacks=[checkpoint]
)

# Save the final model
model.save('oral_disease_model.keras')

