#CNN
# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model
# import tkinter as tk
# from tkinter import messagebox
# import time
#
# # Load the trained Keras model for emotion detection
# model = load_model("cnn_model.h5")
#
# # Load the face cascade classifier
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
#
# # Start video capture
# cap = cv2.VideoCapture(0)
#
# emotions = ['angry', 'stress', 'fear', 'happy', 'sad', 'surprise', 'neutral']
#
# # Initialize variables for tracking popup message display
# last_popup_time = time.time()
# popup_interval = 5  # Set the interval between popup messages (in seconds)
#
# # Function to display popup message
# def display_popup_message(emotion):
#     root = tk.Tk()
#     root.withdraw()  # Hide the main window
#     if emotion in ['stress', 'sad', 'fear']:
#         messagebox.showinfo("Emotion Detected", f"You seem {emotion}. Try relaxing or doing something you enjoy.")
#     elif emotion == 'happy':
#         messagebox.showinfo("Emotion Detected", "You seem happy! Keep up the good mood.")
#     # You can add more conditions for other emotions if needed
#     root.destroy()
#
# while True:
#     ret, frame = cap.read()  # Capture frame
#     if not ret:
#         continue
#
#     # Convert frame to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # Detect faces in the frame
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#
#     # Iterate over detected faces
#     for (x, y, w, h) in faces:
#         # Extract the region of interest (ROI) containing the face
#         roi_gray = gray[y:y + h, x:x + w]
#
#         # Resize the ROI to match the input shape of the model (48x48)
#         roi_gray_resized = cv2.resize(roi_gray, (48, 48))
#
#         # Convert the resized ROI to a format suitable for prediction
#         roi_for_prediction = np.expand_dims(roi_gray_resized, axis=-1)
#         roi_for_prediction = np.expand_dims(roi_for_prediction, axis=0)
#         roi_for_prediction = roi_for_prediction / 255.0  # Normalize pixel values
#
#         # Predict the emotion using the loaded Keras model
#         predictions = model.predict(roi_for_prediction)
#         predicted_emotion = emotions[np.argmax(predictions)]
#
#         # Display the predicted emotion
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
#         cv2.putText(frame, predicted_emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
#
#         # Check if enough time has passed since the last popup message
#         if time.time() - last_popup_time >= popup_interval:
#             # Display popup message if specific emotions are detected
#             if predicted_emotion in ['stress', 'sad', 'fear', 'happy']:
#                 display_popup_message(predicted_emotion)
#                 # Update the last popup time
#                 last_popup_time = time.time()
#
#     # Display the frame
#     cv2.imshow('Emotion Detection', frame)
#
#     # Exit if 'q' key is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Release the video capture object and close all windows
# cap.release()
# cv2.destroyAllWindows()


#---------------------------------------------------------------------------------------------------------------------------------------
#ResNet

# import cv2
# import numpy as np
# from collections import Counter
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array
# from tkinter import messagebox
# import time
#
# # Load the trained Keras model for emotion detection
# model = load_model("emotion_detection_model_resnet.h5")
#
# # Load the face cascade classifier
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
#
# # Define emotions globally
# emotions = ['angry', 'stress', 'fear', 'happy', 'sad', 'surprise', 'neutral']
#
# # Initialize a global variable to store emotion counts
# emotion_counts = Counter()
#
# # Function to capture video and analyze emotions
# def capture_and_analyze():
#     global emotion_counts
#     cap = cv2.VideoCapture(0)
#     last_popup_time = time.time()  # Store the time of the last popup
#
#     while True:
#         ret, frame = cap.read()  # Capture frame
#         if not ret:
#             continue
#
#         # Convert frame to RGB (model expects RGB images)
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#         # Detect faces in the frame
#         faces = face_cascade.detectMultiScale(frame_rgb, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#
#         # Iterate over detected faces
#         for (x, y, w, h) in faces:
#             # Extract the region of interest (ROI) containing the face
#             roi = frame_rgb[y:y + h, x:x + w]
#
#             # Resize the ROI to match the input shape of the model (48x48 for your ResNet model)
#             roi_resized = cv2.resize(roi, (48, 48))
#
#             # Convert the resized ROI to a format suitable for prediction
#             roi_for_prediction = img_to_array(roi_resized)
#             roi_for_prediction = np.expand_dims(roi_for_prediction, axis=0)
#             roi_for_prediction = roi_for_prediction / 255.0  # Normalize pixel values
#
#             # Predict the emotion using the loaded Keras model
#             predictions = model.predict(roi_for_prediction)
#             predicted_emotion = emotions[np.argmax(predictions)]
#
#             # Increment the count for the predicted emotion
#             emotion_counts[predicted_emotion] += 1
#
#             # Check if it's time to display the popup message (every 5 seconds)
#             if time.time() - last_popup_time >= 5:
#                 # Analyze the most observed emotion
#                 most_observed_emotion = emotion_counts.most_common(1)[0][0]
#                 # Show popup message based on the most observed emotion
#                 if most_observed_emotion in ['stress', 'sad', 'fear']:
#                     popup_message = f"You seem {most_observed_emotion}. Try relaxing or doing something you enjoy."
#                 else:  # Assuming 'happy' or other positive emotions
#                     popup_message = "You seem happy! Keep up the good mood."
#                 # Display the popup message
#                 messagebox.showinfo("Emotion Detection", popup_message)
#                 # Update the last popup time
#                 last_popup_time = time.time()
#                 # Clear emotion counts after showing popup
#                 emotion_counts.clear()
#
#             # Display the predicted emotion
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
#             cv2.putText(frame, predicted_emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
#
#         # Display the frame
#         cv2.imshow('Emotion Detection', frame)
#
#         # Break the loop if 'q' key is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     # Release the video capture object
#     cap.release()
#     cv2.destroyAllWindows()
#
# # Start capturing and analyzing video
# capture_and_analyze()

#------------------------------------------------------------------------------------------------------------------------------------------
#VGG
import cv2
import numpy as np
from collections import Counter
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tkinter import messagebox
import time

# Load the trained Keras model for emotion detection
model = load_model("emotion_detection_model_vgg.h5")

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Define emotions globally
emotions = ['angry', 'stress', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Initialize a global variable to store emotion counts
emotion_counts = Counter()

# Function to capture video and analyze emotions
def capture_and_analyze():
    global emotion_counts
    cap = cv2.VideoCapture(0)
    last_popup_time = time.time()  # Store the time of the last popup

    while True:
        ret, frame = cap.read()  # Capture frame
        if not ret:
            continue

        # Convert frame to RGB (model expects RGB images)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(frame_rgb, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Iterate over detected faces
        for (x, y, w, h) in faces:
            # Extract the region of interest (ROI) containing the face
            roi = frame_rgb[y:y + h, x:x + w]

            # Resize the ROI to match the input shape of the model (48x48 for your ResNet model)
            roi_resized = cv2.resize(roi, (48, 48))

            # Convert the resized ROI to a format suitable for prediction
            roi_for_prediction = img_to_array(roi_resized)
            roi_for_prediction = np.expand_dims(roi_for_prediction, axis=0)
            roi_for_prediction = roi_for_prediction / 255.0  # Normalize pixel values

            # Predict the emotion using the loaded Keras model
            predictions = model.predict(roi_for_prediction)
            predicted_emotion = emotions[np.argmax(predictions)]

            # Increment the count for the predicted emotion
            emotion_counts[predicted_emotion] += 1

            # Check if it's time to display the popup message (every 5 seconds)
            if time.time() - last_popup_time >= 5:
                # Analyze the most observed emotion
                most_observed_emotion = emotion_counts.most_common(1)[0][0]
                # Show popup message based on the most observed emotion
                if most_observed_emotion in ['stress', 'sad', 'fear']:
                    popup_message = f"You seem {most_observed_emotion}. Try relaxing or doing something you enjoy."
                else:  # Assuming 'happy' or other positive emotions
                    popup_message = "You look in a good mood! Keep up the good mood."
                # Display the popup message
                messagebox.showinfo("Emotion Detection", popup_message)
                # Update the last popup time
                last_popup_time = time.time()
                # Clear emotion counts after showing popup
                emotion_counts.clear()

            # Display the predicted emotion
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, predicted_emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow('Emotion Detection', frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object
    cap.release()
    cv2.destroyAllWindows()

# Start capturing and analyzing video
capture_and_analyze()