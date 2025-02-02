import streamlit as st
from openai import OpenAI
import tempfile
import os
from moviepy.editor import VideoFileClip

# Sidebar for API Key input
st.sidebar.title("Get started")
api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")

# Main app title
st.title("🎙️ Speech2Text with Whisper-1 🤖")

# Upload file (audio or video)
uploaded_file = st.file_uploader(
    "Upload an audio or video file", 
    type=["mp3", "wav", "mp4", "mov", "avi", "mpeg", "m4a"]
)

# Check if API key is provided
if not api_key:
    st.warning("Please enter your OpenAI API key to proceed.")
    st.stop()

# Initialize OpenAI client
try:
    client = OpenAI(api_key=api_key)
except Exception as e:
    st.error(f"Invalid API key or OpenAI initialization failed: {e}")
    st.stop()

# Process file when user clicks "Convert"
if uploaded_file is not None:
    st.write(f"Uploaded file: {uploaded_file.name}")
    
    # Save the uploaded file temporarily
    file_extension = uploaded_file.name.split('.')[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # Button to start transcription
    if st.button("Convert"):
        with st.spinner("Processing... Please wait."):
            try:
                # Handle video files
                if file_extension in ["mp4", "mov", "avi", "mpeg"]:
                    st.write("Extracting audio from video...")

                    video_clip = VideoFileClip(tmp_file_path)
                    if video_clip.audio is None:
                        st.error("The uploaded video does not have an audio stream.")
                        st.stop()

                    # Extract and save the audio as MP3
                    audio_path = tmp_file_path.rsplit('.', 1)[0] + ".mp3"
                    video_clip.audio.write_audiofile(audio_path)
                    video_clip.close()

                    # Transcribe the extracted audio
                    with open(audio_path, "rb") as audio_file:
                        transcription_response = client.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio_file
                        )
                    os.remove(audio_path)

                # Handle audio files
                elif file_extension in ["mp3", "wav", "m4a"]:
                    with open(tmp_file_path, "rb") as audio_file:
                        transcription_response = client.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio_file
                        )

                else:
                    st.error("Unsupported file type uploaded.")
                    st.stop()

                # Display the transcription text
                transcription_text = transcription_response.get('text', "No transcription available.")
                st.success("Transcription completed!")
                st.write("**Transcription:**")
                st.text(transcription_text)

            except Exception as e:
                st.error(f"An error occurred during transcription: {e}")

            finally:
                # Remove the temporary file
                os.remove(tmp_file_path)

else:
    st.info("Upload an audio or video file to start.")
