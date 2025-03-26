# Import dependencies
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from scipy.io.wavfile import write as writeaudio
import mediapipe as mp 
import cv2  
import numpy as np
import pandas as pd
import pickle
import threading
import warnings
import torch
from transformers import pipeline
import google.generativeai as genai
from elevenlabs import play
from elevenlabs.client import ElevenLabs
import sounddevice as sd
warnings.filterwarnings("ignore")

# Backend initialization
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Facial emotion detection initialization
mp_holistic = mp.solutions.holistic  # Mediapipe Solutions
emotion = None

# Load the body language model
with open('emotion_model.pkl', 'rb') as f:
    body_language_model = pickle.load(f)

# LLM initialization
genai.configure(api_key="")

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

chat_session = model.start_chat(
    history=[]
)

# TTS initialization
EL_API_KEY = ''
client = ElevenLabs(api_key=EL_API_KEY)

# Handling API calls
@app.route('/api/call', methods=['POST'])
def handle_transcript():
    global emotion
    data = request.json
    transcript = data.get('transcript')
    print("User: ", transcript)
    answer = chat(transcript, emotion)
    print("Pancake: ", answer)
    #readaloudEL(answer)
    return jsonify({"message": answer}), 200

@app.route('/process-image', methods=['POST'])
def process_image():
    global emotion
    # Get the image from the request
    data = request.files['image'].read()
    np_image = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    # Process the image using MediaPipe
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False
        
        # Make Detections
        results = holistic.process(img_rgb)

        # Initialize lists for landmarks
        pose_row, face_row, hands_row = [], [], []

        # Check for pose landmarks
        if results.pose_landmarks:
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

        # Check for face landmarks
        if results.face_landmarks:
            face = results.face_landmarks.landmark
            face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())

        # Check for right hand landmarks
        if results.right_hand_landmarks:
            right_hand = results.right_hand_landmarks.landmark
            hands_row.extend(list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in right_hand]).flatten()))

        # Check for left hand landmarks
        if results.left_hand_landmarks:
            left_hand = results.left_hand_landmarks.landmark
            hands_row.extend(list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in left_hand]).flatten()))

        # Combine all rows
        combined_row = pose_row + face_row + hands_row

        # Make Detections if any landmarks were found
        if combined_row:
            X = pd.DataFrame([combined_row])
            body_language_class = body_language_model.predict(X)[0]  # Get the classification result
            print("Classification Result:", body_language_class)
            emotion = body_language_class

            return jsonify({"classification": body_language_class}), 200
        else:
            return jsonify({"error": "No landmarks detected"}), 400

# LLM functions
def chat(user_input, emotion):
    user_input = f"User facial emotion: {emotion}. Prompt: {user_input}. As a therapist, respond to the user, taking into account their facial emotion. Respond as a text."
    response = chat_session.send_message(user_input)
    return response.text

# TTS functions
def readaloudEL(text_input):
    audio = client.generate(
        text=text_input,
        voice="Chris",
        model="eleven_multilingual_v2"
    )
    play(audio)

if __name__ == '__main__':
    app.run(debug=True, port=8080)
