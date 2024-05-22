import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import CSVLogger
from data_preprocessing import train_generator, test_generator

# Load the pre-trained MobileNet model without the top layer
mobilenet_base = MobileNet(weights='imagenet', include_top=False, input_shape=(48, 48, 3))

# Create a new model based on MobileNet
mobilenet_model = Sequential()

# Add the pre-trained MobileNet base model
mobilenet_model.add(mobilenet_base)

# Add a global average pooling layer
mobilenet_model.add(GlobalAveragePooling2D())

# Add a dense output layer with 7 units (for 7 classes) and softmax activation
mobilenet_model.add(Dense(7, activation='softmax'))

# Compile the model
mobilenet_model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.001, decay=1e-6), metrics=['accuracy', Precision(), Recall()])

# Set up CSV logger
csv_logger = CSVLogger('mobilenet_training.log')

# Train the model
mobilenet_model_info = mobilenet_model.fit(
        train_generator,
        steps_per_epoch=28709 // 64,
        epochs=10,
        validation_data=train_generator,
        validation_steps=7178 // 64,
        callbacks=[csv_logger])

mobilenet_model.save('emotion_detection_model_mobilenet.h5')

# Evaluate the model
train_loss, train_accuracy, train_precision, train_recall = mobilenet_model.evaluate(train_generator, verbose=0)
print("Training Accuracy:", train_accuracy)

test_loss, test_accuracy, test_precision, test_recall = mobilenet_model.evaluate(test_generator, verbose=0)
print("Test Accuracy:", test_accuracy)
