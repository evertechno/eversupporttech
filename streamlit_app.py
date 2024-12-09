import streamlit as st
import whisper
from transformers import pipeline
import pandas as pd
import io

# Function to load Whisper model
def load_whisper_model():
    model = whisper.load_model("base")  # You can choose different model sizes
    return model

# Function to convert audio file to text using Whisper
def audio_to_text(audio_file):
    model = load_whisper_model()
    # Save the uploaded file as a temporary file
    with open("temp.wav", "wb") as f:
        f.write(audio_file.getbuffer())

    # Transcribe the audio file to text
    result = model.transcribe("temp.wav")
    return result['text']

# Function to analyze the transcript with Hugging Face Transformers (e.g., summarization, score generation)
def analyze_transcript(text):
    summarizer = pipeline("summarization")
    score_generator = pipeline("text-classification", model="facebook/bart-large-mnli")  # You can replace with your custom model for scoring

    # Generate summary
    summary = summarizer(text, max_length=200, min_length=50, do_sample=False)

    # Generate support score (can be customized to more specific scoring logic)
    score = score_generator(text)

    return summary[0]['summary_text'], score[0]['label'], score[0]['score']

# Streamlit App Interface
def main():
    st.title("Call Recording Analysis")

    # Upload audio file (only WAV format)
    audio_file = st.file_uploader("Upload Call Recording (WAV format)", type=["wav"])

    if audio_file is not None:
        # Display uploaded file name
        st.write(f"Uploaded file: {audio_file.name}")

        # Convert audio to text
        with st.spinner("Converting audio to text..."):
            transcript = audio_to_text(audio_file)
        
        st.subheader("Transcript")
        st.write(transcript)  # Show the transcript of the call

        # Analyze the transcript (summarization and score)
        with st.spinner("Analyzing the call..."):
            summary, score_label, score_value = analyze_transcript(transcript)
        
        st.subheader("Call Summary")
        st.write(summary)  # Show the summary of the call
        
        st.subheader("Agent Performance Score")
        st.write(f"Score Label: {score_label}, Score Value: {score_value}")

        # Display some basic metrics (for example, word count, speaking time, etc.)
        word_count = len(transcript.split())
        st.write(f"Word Count: {word_count}")

        # Optionally, you can create a DataFrame to display metrics
        metrics = {
            "Word Count": [word_count],
            "Summary": [summary],
            "Score": [score_label],
            "Score Value": [score_value]
        }

        metrics_df = pd.DataFrame(metrics)
        st.subheader("Agent Metrics")
        st.dataframe(metrics_df)

if __name__ == "__main__":
    main()
