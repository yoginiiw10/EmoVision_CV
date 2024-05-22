import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger
from data_preprocessing import train_generator, test_generator

# Load the pre-trained ResNet50 model without the top layer
resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=(48, 48, 3))

# Create a new model based on ResNet50
resnet_model = Sequential()

# Add the pre-trained ResNet50 base model
resnet_model.add(resnet_base)

# Add a global average pooling layer
resnet_model.add(GlobalAveragePooling2D())

# Add a dense output layer with 7 units (for 7 classes) and softmax activation
resnet_model.add(Dense(7, activation='softmax'))

# Compile the model
resnet_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001, decay=1e-6), metrics=['accuracy'])

# Set up CSV logger
csv_logger = CSVLogger('resnet_training.log')

# Train the model
resnet_model_info = resnet_model.fit(
        train_generator,
        steps_per_epoch=28709 // 64,
        epochs=10,
        validation_data=train_generator,
        validation_steps=7178 // 64,
        callbacks=[csv_logger])

resnet_model.save('emotion_detection_model_resnet.h5')

# Evaluate the model
Y_true = test_generator.classes
Y_pred = resnet_model.predict(test_generator)
Y_pred_classes = np.argmax(Y_pred, axis=1)
