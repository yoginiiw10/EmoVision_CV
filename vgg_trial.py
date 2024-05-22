import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger
from data_preprocessing import train_generator, test_generator

# Load the pre-trained VGG16 model without the top layer
vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(48, 48, 3))

# Create a new model based on VGG16
vgg_model = Sequential()

# Add the pre-trained VGG16 base model
vgg_model.add(vgg_base)

# Add a global average pooling layer
vgg_model.add(GlobalAveragePooling2D())

# Add a dense output layer with 7 units (for 7 classes) and softmax activation
vgg_model.add(Dense(7, activation='softmax'))

# Compile the model with adjusted hyperparameters
vgg_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.00001, decay=1e-6), metrics=['accuracy'])

# Set up CSV logger
csv_logger = CSVLogger('VGG_TRAIN.log')

# Train the model with adjusted number of epochs
vgg_model_info = vgg_model.fit(
        train_generator,
        steps_per_epoch=28709 // 64,
        epochs=20,  # Increased number of epochs
        validation_data=test_generator,
        validation_steps=7178 // 64,
        callbacks=[csv_logger])

vgg_model.save('emoVGG.h5')

# Evaluate the model
Y_true = test_generator.classes
Y_pred = vgg_model.predict(test_generator)
Y_pred_classes = np.argmax(Y_pred, axis=1)
