import tkinter as tk
import threading
import pyttsx3
import speech_recognition as sr
import wikipedia
import datetime
import webbrowser
import os
from PIL import Image, ImageTk
import cv2
import numpy as np
import torch
import tensorflow as tf
from pathlib import Path

# Initialize pyttsx3 engine
engine = pyttsx3.init()

# Adjust properties if needed
engine.setProperty('rate', 150)  # Speed of speech, adjust as necessary
engine.setProperty('volume', 0.9)  # Volume level, adjust as necessary

def speak(text):
    engine.say(text)
    engine.runAndWait()
    
# Load currency classification model
currency_model = tf.keras.models.load_model("currency_classification_model.h5")
class_labels = {0: 'Ten Rupee Notes', 1: 'Hundred Rupee Notes', 2: 'Twenty Rupee Notes',
                3: 'Two Hundred Rupee Notes', 4: 'Fifty Rupee Notes',
                5: 'Five Hundred Rupee Notes'}

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5:v7.0', 'yolov5m', pretrained=True)
model.eval()

# Function to convert speech to text
def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        speak("Listening...")  # Move "Listening..." prompt here to reduce delay
        try:
            audio = recognizer.listen(source, timeout=5)  # Set timeout to 5 seconds
        except sr.WaitTimeoutError:
            print("Timeout reached. No audio input received.")
            return None
    try:
        print("Recognizing...")
        command = recognizer.recognize_google(audio).lower()
        print("Command:", command)
        return command
    except sr.UnknownValueError:
        print("Could not understand audio.")
        speak("Sorry, I didn't catch that. Please try again.")
        return None
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))
        speak("Sorry, I encountered an error. Please try again later.")
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


# Function to speak the detection results
def speak_detection(label, location, confidence):
    text = f'{label}: Location - {location}, Confidence - {confidence:.2f}'
    print(text)  # Print the detection result as before
    speak(text)

def speak_detection2(label):
    text = f'{label}'
    print(text)  # Print the detection result as before
    speak(text)

# Tkinter GUI
class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Visual Assistant A.I.")
        self.configure(bg='black')  # Set background color to black

        # Load the background image
        self.background_image = Image.open("back.png")
        self.background_image = self.background_image.resize((self.winfo_screenwidth(), self.winfo_screenheight()))  # Resize the image to fit the screen
        self.background_photo = ImageTk.PhotoImage(self.background_image)

        # Create a canvas to display the background image
        self.canvas = tk.Canvas(self, width=self.winfo_screenwidth(), height=self.winfo_screenheight(), bg='black', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.background_photo)

        # Create a frame to contain the label
        self.label_frame = tk.Frame(self, bg='black')
        self.label_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Create the label
        self.label = tk.Label(self.label_frame, text="Visual Assistant A.I. Click anywhere to continue", font=("Arial", 36), fg='white', bg='black')  # Set text color to white
        self.label.pack()

        self.bind("<Button-1>", self.start_assistant)

    def start_assistant(self, event):
        self.label.config(text="Visual Assistant A.I. has Started", font=("Arial", 36))
        threading.Thread(target=self.run_assistant).start()

    def run_assistant(self):
        greet()
        while True:
            query = listen()  # Removed the "Listening..." prompt here to reduce delay
            if query is not None:
                if 'wikipedia' in query:
                    speak('Searching Wikipedia...')
                    query = query.replace("wikipedia", "")
                    try:
                        results = wikipedia.summary(query, sentences=2)
                        speak("According to Wikipedia")
                        print(results)
                        speak(results)
                    except wikipedia.exceptions.DisambiguationError:
                        speak("Multiple results found. Please specify your query.")
                    except wikipedia.exceptions.PageError:
                        speak("No matching page found.")
                    except wikipedia.exceptions.WikipediaException as e:
                        speak(f"An unknown error occurred: {e}. Please try again later.")
                elif 'open youtube' in query:
                    speak("Opening Youtube sir")
                    os.system("start https://www.youtube.com")
                elif 'open google' in query:
                    speak("Opening Google sir")
                    webbrowser.open("https://www.google.com")
                elif 'time' in query:
                    hour = datetime.datetime.now().strftime("%H")
                    min = datetime.datetime.now().strftime("%M")
                    speak(f"The time is {hour} {min}")
                elif 'climate' in query:
                    speak("so todays weather is as follows: ")
                    os.system("start https://www.google.com/search?q=weather+of+today&gs_ivs=1#tts=0")
                elif 'bye' in query:
                    speak("Goodbye! and take care")
                    break
                elif 'object' in query:
                    process_object_detection(confidence_threshold=0.6)
                elif 'currency' in query:
                    process_currency_detection()
                elif 'stop' in query:
                    break
                else:
                    speak("Sorry, I couldn't understand that sir")
        self.label.config(text="Click anywhere to start Visual Assistant A.I.", font=("Arial", 24))

def greet():
    hour = int(datetime.datetime.now().hour)
    if hour >= 0 and hour < 12:
        speak("Good Morning!")
    elif hour >= 12 and hour < 18:
        speak("Good Afternoon!")
    else:
        speak("Good Evening!")
    speak("I am Visual Assistant A.I. How may I assist you?")

if __name__ == "__main__":
    app = Application()
    app.mainloop()
