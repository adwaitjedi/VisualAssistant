from flask import Flask, render_template
from threading import Thread
import cv2
import numpy as np
import pyttsx3
import speech_recognition as sr
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

# Load currency classification model
currency_model = tf.keras.models.load_model("currency_classification_model.h5")
class_labels = {0: 'Ten Rupee Notes', 1: 'Hundred Rupee Notes', 2: 'Twenty Rupee Notes',
                3: 'Two Hundred Rupee Notes', 4: 'Fifty Rupee Notes',
                5: 'Five Hundred Rupee Notes'}

# Initialize pyttsx3 engine
engine = pyttsx3.init()

# Adjust properties if needed
engine.setProperty('rate', 150)  # Speed of speech, adjust as necessary
engine.setProperty('volume', 0.9)  # Volume level, adjust as necessary

# Function to speak the detection results
def speak_detection(label, location, confidence):
    text = f'{label}: Location - {location}, Confidence - {confidence:.2f}'
    print(text)  # Print the detection result as before
    engine.say(text)
    engine.runAndWait()

# Define image processing pipeline for object detection with OpenCV
def process_object_detection(confidence_threshold=0.5):
    # Open the default camera (usually webcam)
    cap = cv2.VideoCapture(0)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Unable to open camera.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If frame is read correctly, ret is True
        if not ret:
            print("Error: Unable to capture frame.")
            break

        # Perform inference with your object detection model here
        # Replace this section with your actual object detection code
        # For example, you can use YOLOv5 model here
        # Your object detection code goes here

        # Display the resulting frame
        cv2.imshow('Frame', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture
    cap.release()
    cv2.destroyAllWindows()

# Define image processing pipeline for currency detection
def process_currency_detection():
    # Open the default camera (usually webcam)
    cap = cv2.VideoCapture(0)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Unable to open camera.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If frame is read correctly, ret is True
        if not ret:
            print("Error: Unable to capture frame.")
            break

        # Currency Detection
        img = Image.fromarray(frame)
        img = img.resize((150, 150))  # Resize the image to match the input size expected by the model
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Add an extra dimension for the batch
        predictions = currency_model.predict(img_array)
        predicted_class = np.argmax(predictions)
        predicted_label = class_labels[predicted_class]
        print(f'The predicted currency is: {predicted_label}')

        # Display the resulting frame
        cv2.imshow('Frame', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture
    cap.release()
    cv2.destroyAllWindows()

# Route for the index page
@app.route('/')
def index():
    return render_template('index.html')

# Route for starting object detection
@app.route('/start_object_detection')
def start_object_detection():
    Thread(target=process_object_detection).start()
    return "Object detection started."

# Route for starting currency detection
@app.route('/start_currency_detection')
def start_currency_detection():
    Thread(target=process_currency_detection).start()
    return "Currency detection started."

if __name__ == '__main__':
    app.run(debug=True)
