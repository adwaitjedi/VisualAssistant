import torch
from pathlib import Path
import cv2
from matplotlib import pyplot as plt
import numpy as np
import pyttsx3
import speech_recognition as sr
from PIL import Image
import tensorflow as tf

# Load currency classification model
currency_model = tf.keras.models.load_model("currency_classification_model.h5")
class_labels = {0: 'Ten Rupee Notes', 1: 'Hundred Rupee Notes', 2: 'Twenty Rupee Notes',
                3: 'Two Hundred Rupee Notes', 4: 'Fifty Rupee Notes',
                5: 'Five Hundred Rupee Notes'}

# YOLOv5 repository path (relative to the script)
yolov5_path = Path("folder/yolov5")

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5:v7.0', 'yolov5m', pretrained=True)

# Set the model in evaluation mode
model.eval()

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

def speak_detection2(label):
    text = f'{label}'
    print(text)  # Print the detection result as before
    engine.say(text)
    engine.runAndWait()

# Function to convert speech to text
def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source)
    try:
        print("Recognizing...")
        command = recognizer.recognize_google(audio).lower()
        print("Command:", command)
        return command
    except sr.UnknownValueError:
        print("Could not understand audio.")
        return None
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))
        return None

# Define image processing pipeline for object detection with YOLOv5
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

        # Perform inference
        results = model(frame)

        # Process detection results
        for detection in results.xyxy[0]:
            label = model.names[int(detection[5])]
            confidence = float(detection[4])

            # Skip detection if confidence is below the threshold
            if confidence < confidence_threshold:
                continue

            # Get bounding box coordinates
            bbox = detection[0:4].cpu().numpy().astype(int)  # Adjusted index and added type conversion

            # Calculate center coordinates
            center_x, center_y = (bbox[2] + bbox[0]) // 2, (bbox[3] + bbox[1]) // 2

            # Define location based on center coordinates
            location = ""

            if center_y < frame.shape[0] // 3:
                location += "Top"
            elif center_y > 2 * frame.shape[0] // 3:
                location += "Bottom"
            else:
                location += "Center"

            if center_x < frame.shape[1] // 3:
                location += "-Left"
            elif center_x > 2 * frame.shape[1] // 3:
                location += "-Right"

            # Display the result
            print(f'{label}: Location - {location}, Confidence - {confidence:.2f}')
            speak_detection(label, location, confidence)

            # Draw bounding box and label on the image
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(frame, f'{label}: {confidence:.2f}', (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Frame', frame)

        # Break the loop if 'q' is pressed or command is received
        if cv2.waitKey(1) & 0xFF == ord('q') or listen() == "stop":
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
        speak_detection2(predicted_label)


        # Display the resulting frame
        cv2.imshow('Frame', frame)

        # Break the loop if 'q' is pressed or command is received
        if cv2.waitKey(1) & 0xFF == ord('q') or listen() == "stop":
            break

    # Release the capture
    cap.release()
    cv2.destroyAllWindows()


stop_key=["stop","stop kr na","stop the camera"]
start_object_detection_key=["start","start object detection now","start object detection please"]
start_currency_detection_key=["begin","start currency detection now","start currency detection please"]

# Example usage with custom threshold
if __name__ == '__main__':
    while True:
        x = listen()
        if x and x.lower() in start_object_detection_key:
            process_object_detection(confidence_threshold=0.6)
        elif x and x.lower() in start_currency_detection_key:
            process_currency_detection()
        elif x in stop_key:
            break
