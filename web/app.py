from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import scipy.io.wavfile as wav
import librosa
from sklearn.externals import joblib

app = Flask(__name__)

# Load your pre-trained classifier
classifier = joblib.load('your_pretrained_classifier.pkl')

def detect_voice(audio_file):
    # Process the audio file, extract features, and pass them to the classifier
    # Return the result as "Human" or "AI"
    sr, audio = wav.read(audio_file)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    features = np.hstack((mfcc_mean, mfcc_std))
    result = classifier.predict([features])

    if result[0] == 0:
        return "Human"
    else:
        return "AI"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    audio_file = request.files['audio_file']
    audio_file.save = ('audio.wav')