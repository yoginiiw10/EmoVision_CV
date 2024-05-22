from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from data_preprocessing import train_generator, test_generator


# Define the model
# Define the model with increased complexity
def EmotionNetModel(input_shape=(224, 224, 3), num_classes=7):
    base_model = MobileNet(input_shape=input_shape, include_top=False, weights='imagenet')

    for layer in base_model.layers[:-10]:  # Fine-tuning more layers
        layer.trainable = True

    x = Flatten()(base_model.output)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)  # Increased units
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)  # Additional dense layer
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# Create the model
model = EmotionNetModel()

# Compile the model with reduced learning rate
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Set up callbacks
checkpoint = ModelCheckpoint("emotion_detection_model.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_accuracy', patience=8, verbose=1, mode='max', restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001, verbose=1)

# Train the model
history = model.fit(train_generator, epochs=50, validation_data=test_generator, callbacks=[checkpoint, early_stopping, reduce_lr])

# Print training and validation accuracy
print("Training Accuracy:", history.history['accuracy'])
print("Validation Accuracy:", history.history['val_accuracy'])