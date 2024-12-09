import streamlit as st
import numpy as np
import pandas as pd
from deepspeech import Model
import wave
import os
from transformers import pipeline
import urllib.request
import subprocess
import sys

# Set up Hugging Face for text summarization and sentiment analysis
summarizer = pipeline("summarization")
sentiment_analyzer = pipeline("sentiment-analysis")

# Define URLs for DeepSpeech model and scorer
DEEPSPEECH_MODEL_URL = 'https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm'
DEEPSPEECH_SCORER_URL = 'https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer'

# Model paths
DEEPSPEECH_MODEL_PATH = 'deepspeech-0.9.3-models.pbmm'
DEEPSPEECH_SCORER_PATH = 'deepspeech-0.9.3-models.scorer'

# Function to download the model if it doesn't exist
def download_model(model_url, model_path):
    if not os.path.exists(model_path):
        st.write(f"Downloading model from {model_url}...")
        urllib.request.urlretrieve(model_url, model_path)
        st.write("Download complete!")

# Ensure 'librosa' is installed if not already
def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import librosa
except ImportError:
    install_package("librosa")
    import librosa

def load_deepspeech_model():
    # Ensure the model is downloaded
    download_model(DEEPSPEECH_MODEL_URL, DEEPSPEECH_MODEL_PATH)
    download_model(DEEPSPEECH_SCORER_URL, DEEPSPEECH_SCORER_PATH)
    
    model = Model(DEEPSPEECH_MODEL_PATH)
    model.enableExternalScorer(DEEPSPEECH_SCORER_PATH)
    return model

def transcribe_audio(model, audio_file):
    # Load audio file with librosa
    audio_data, _ = librosa.load(audio_file, sr=16000)
    
    # Save to a temporary wav file
    temp_filename = 'temp_audio.wav'
    librosa.output.write_wav(temp_filename, audio_data, 16000)
    
    # Read the temporary wav file
    with wave.open(temp_filename, 'rb') as f:
        frames = f.readframes(f.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16)
    
    # Perform speech-to-text conversion
    text = model.stt(audio)
    os.remove(temp_filename)  # Clean up temporary file
    return text

def analyze_text(text):
    # Summarize the transcript
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
    
    # Sentiment analysis
    sentiment = sentiment_analyzer(text)
    
    # Simulate a score (e.g., a performance score for agents)
    score = np.random.randint(70, 100)  # Placeholder for a more complex metric
    
    return summary, sentiment, score

def display_results(summary, sentiment, score):
    st.subheader("Call Summary")
    st.write(summary)
    
    st.subheader("Sentiment Analysis")
    st.write(f"Sentiment: {sentiment[0]['label']}, Confidence: {sentiment[0]['score']*100:.2f}%")
    
    st.subheader("Agent Performance Score")
    st.write(f"Performance Score: {score}/100")

def main():
    st.title("Call Audit Analysis Web App")
    st.sidebar.header("Upload Call Recording")
    
    uploaded_file = st.sidebar.file_uploader("Choose an audio file", type=["wav", "mp3", "flac"])
    
    if uploaded_file:
        st.audio(uploaded_file, format='audio/wav')
        st.write("Processing your file...")
        
        # Initialize the DeepSpeech model
        model = load_deepspeech_model()
        
        # Convert audio to text
        transcript = transcribe_audio(model, uploaded_file)
        st.subheader("Transcript")
        st.write(transcript)
        
        # Perform analysis (summary, sentiment, score)
        summary, sentiment, score = analyze_text(transcript)
        
        # Display results
        display_results(summary, sentiment, score)

if __name__ == "__main__":
    main()
