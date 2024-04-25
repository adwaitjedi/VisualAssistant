from flask import Flask, render_template, request, jsonify
import subprocess
import json
import spacy
import os

import torch
from pathlib import Path
import cv2
from matplotlib import pyplot as plt
import numpy as np
import pyttsx3
import speech_recognition as sr
from PIL import Image
import tensorflow as tf

app = Flask(__name__)





@app.route('/')
def index():
    
    return render_template('chat.html')

@app.route('/detect', methods=['POST'])
def detect():
    speak_detection()



    # Get user-entered classes
    user_input = request.form.get('user_input')

    # Extract keywords using spaCy
    

if __name__ == '__main__':
    app.run(debug=True)
