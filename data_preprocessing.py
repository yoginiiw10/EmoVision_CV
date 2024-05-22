import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = "C:\\Users\\yogin\\Desktop\\EmoVision\\data\\train"
test_dir = "C:\\Users\\yogin\\Desktop\\EmoVision\\data\\test"

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=64,
    class_mode='categorical',
    color_mode='rgb'  # Ensure images are loaded with three channels (RGB)
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    batch_size=64,
    class_mode='categorical',
    color_mode='rgb'  # Ensure images are loaded with three channels (RGB)
)
