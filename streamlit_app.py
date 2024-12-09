import streamlit as st
import whisper
from pyannote.audio import Pipeline
import google.generativeai as genai
import os
import urllib.request

# Configure the API key securely from Streamlit's secrets
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Initialize Whisper model
model = whisper.load_model("base")  # You can choose from "tiny", "base", "small", "medium", "large"

# Initialize speaker diarization pipeline (will automatically download models)
diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

def transcribe_audio_whisper(audio_file):
    result = model.transcribe(audio_file)
    return result['text']

def diarize_audio(audio_file):
    diarization = diarization_pipeline({'uri': 'filename', 'audio': audio_file})
    return diarization

def label_conversation(diarization, transcript):
    labeled_conversation = []
    for segment in diarization.itertracks(yield_label=True):
        start, end, speaker = segment
        segment_text = extract_text_for_segment(transcript, start, end)
        labeled_conversation.append(f"{speaker}: {segment_text}")
    
    return labeled_conversation

def extract_text_for_segment(transcript, start, end):
    return transcript[start:end]

def analyze_text_with_gemini(text):
    prompt = f"Analyze the following text: {text}. Provide a summary, identify key points, and suggest potential insights or actions."
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text

# Streamlit app
st.title("Audio Transcription, Speaker Identification, and AI Analysis")
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.read())

    # Perform speaker diarization
    diarization = diarize_audio("temp_audio.wav")

    # Transcribe audio to text using Whisper
    transcript = transcribe_audio_whisper("temp_audio.wav")
    
    # Label the conversation with the identified speakers
    labeled_conversation = label_conversation(diarization, transcript)
    
    st.write("Labeled Conversation:")
    for line in labeled_conversation:
        st.write(line)

    try:
        # Join labeled conversation into a single text for analysis
        conversation_text = "\n".join(labeled_conversation)
        analysis_result = analyze_text_with_gemini(conversation_text)
        st.write("AI Analysis:")
        st.write(analysis_result)
    except Exception as e:
        st.error(f"Error: {e}")
