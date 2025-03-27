# #Block 1 import libraries
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# from torchvision import transforms
# from transformers import ViTModel
# from PIL import Image
# import numpy as np
# import os
# from sklearn.model_selection import train_test_split
# from tqdm import tqdm
# import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.metrics import MeanIoU
from sklearn.model_selection import train_test_split
import cv2


#Block 2 directories of the dataset
# Define dataset paths
# image_dir = 'C:/Users/Nishith/Downloads/New_train_Patches'
# mask_dir = 'G:/New_VOC_Label_Patches'
IMAGE_DIR = 'C:/Users/Nishith/Downloads/New_train_Patches'  # Update this path
MASK_DIR = 'G:/New_VOC_Label_Patches'   # Update this path
IMG_SIZE = 256  # Image size (256x256)
NUM_CLASSES = 4  # Classes (1,2,3 + Background)
BATCH_SIZE = 16
EPOCHS = 20


# Data generator (Lazy loading to prevent memory issues)
def data_generator(image_dir, mask_dir, batch_size):
    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))
    dataset_size = len(image_files)
    
    def load_sample(img_file, mask_file):
        img = cv2.imread(os.path.join(image_dir, img_file))
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
        
        mask = cv2.imread(os.path.join(mask_dir, mask_file), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
        mask = np.expand_dims(mask, axis=-1)
        mask = keras.utils.to_categorical(mask, num_classes=NUM_CLASSES)
        
        return img, mask
    
    def generator():
        for img_file, mask_file in zip(image_files, mask_files):
            yield load_sample(img_file, mask_file)
    
    dataset = tf.data.Dataset.from_generator(generator, 
                                             output_signature=(
                                                 tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
                                                 tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, NUM_CLASSES), dtype=tf.float32)
                                             ))
    dataset = dataset.shuffle(dataset_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

# Create datasets
train_dataset = data_generator(IMAGE_DIR, MASK_DIR, BATCH_SIZE)
val_dataset = data_generator(IMAGE_DIR, MASK_DIR, BATCH_SIZE)  # Adjust paths for actual validation split

# Load pretrained ViT
import timm
from tensorflow.keras.applications import ResNet50  # Alternative CNN-based model

vit_model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
base_model = tf.keras.Model(inputs=vit_model.default_cfg['input_size'], outputs=vit_model.forward)

# Custom segmentation head
x = base_model.output
x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = layers.Conv2D(NUM_CLASSES, (1, 1), activation='softmax', padding='same')(x)
model = keras.Model(inputs=base_model.input, outputs=x)

# Compile model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss="categorical_crossentropy",
              metrics=[MeanIoU(num_classes=NUM_CLASSES)])


# Train model
history = model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS)

# Save model
model.save("vit_segmentation_model.h5")

# Evaluate
loss, iou = model.evaluate(val_dataset)
print(f"Validation Loss: {loss}, Mean IoU: {iou}")