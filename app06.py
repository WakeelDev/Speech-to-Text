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

# Process button
if uploaded_file is not None:
    st.write(f"Uploaded file: {uploaded_file.name}")
    
    if st.button("Start Processing"):  # Button to start processing
        with st.spinner("Processing... Please wait."):
            file_extension = uploaded_file.name.split('.')[-1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                # Handle video files
                if file_extension in ["mp4", "mov", "avi", "mpeg"]:
                    st.write("Extracting audio from video...")
                    video_clip = VideoFileClip(tmp_file_path)
                    if video_clip.audio is None:
                        st.error("The uploaded video does not have an audio stream.")
                        st.stop()
                    
                    audio_path = tmp_file_path.rsplit('.', 1)[0] + ".mp3"
                    video_clip.audio.write_audiofile(audio_path)
                    video_clip.close()
                    
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
                
                transcription_text = transcription_response.get('text', "No transcription available.")
                st.success("Transcription completed!")
                st.write("**Transcription:**")
                st.text_area("", transcription_text, height=200)
                
                # Download button for transcription
                st.download_button(
                    label="Download Transcription",
                    data=transcription_text,
                    file_name="transcription.txt",
                    mime="text/plain"
                )
            
            except Exception as e:
                st.error(f"An error occurred during transcription: {e}")
            
            finally:
                os.remove(tmp_file_path)
else:
    st.info("Upload an audio or video file to start.")
