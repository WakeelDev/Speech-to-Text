import streamlit as st 
from openai import OpenAI
import tempfile
import os
from moviepy.editor import VideoFileClip

# Sidebar for API Key input
st.sidebar.title("Get started")
api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")

# Main app title
st.title("🎙️Speech2Text with Whisper-1🤖")

# Upload file (audio or video)
uploaded_file = st.file_uploader("Upload an audio or video file", type=["mp3", "wav", "mp4", "mov", "avi"])

# Initialize the OpenAI client with the API key from the sidebar
client = OpenAI(api_key=api_key)

if uploaded_file is not None and api_key:
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix="." + uploaded_file.name.split('.')[-1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        # If the uploaded file is a video, extract audio
        if uploaded_file.name.endswith(('.mp4', '.mov', '.avi')):
            st.write("Extracting audio from video...")

            # Load the video file using moviepy
            video_clip = VideoFileClip(tmp_file_path)
            audio_path = tmp_file_path.rsplit('.', 1)[0] + ".mp3"

            # Extract audio from the video and save it as MP3
            video_clip.audio.write_audiofile(audio_path)

            # Now, use the extracted audio file for transcription
            with open(audio_path, "rb") as audio_file:
                transcription_response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )

            # Clean up: remove temporary video and audio files
            os.remove(audio_path)
        else:
            # If it's an audio file (mp3 or wav), transcribe directly
            with open(tmp_file_path, "rb") as audio_file:
                transcription_response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )

        # Accessing the transcription text correctly
        transcription_text = transcription_response.text

        # Display the transcription
        st.write("Transcription:", transcription_text)

    except Exception as e:
        # Display any errors that occur during transcription
        st.error(f"An error occurred: {str(e)}")

    finally:
        # Clean up: remove the temporary file
        os.remove(tmp_file_path)
