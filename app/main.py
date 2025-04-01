# main.py - Complete Updated Version
import os
import time
import subprocess
import sys
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from gtts import gTTS
import cv2
from PIL import Image
import speech_recognition as sr
from st_audiorec import st_audiorec
from pydub import AudioSegment
import warnings
from google.api_core.client_options import ClientOptions

warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(
    api_key=os.getenv("GOOGLE_API_KEY"),
    client_options=ClientOptions(api_endpoint="generativelanguage.googleapis.com")
)

# Custom CSS
css_code = """
    <style>
    section[data-testid="stSidebar"] > div > div:nth-child(2) {
        padding-top: 0.75rem !important;
    }
    section.main > div {
        padding-top: 64px;
    }
    </style>
"""

# Ensure FFmpeg is installed
def ensure_ffmpeg_installed():
    try:
        subprocess.run(['ffmpeg', '-version'], 
                      check=True, 
                      stdout=subprocess.PIPE, 
                      stderr=subprocess.PIPE)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        st.warning("FFmpeg not found. Trying to install...")
        try:
            if sys.platform == 'linux':
                subprocess.run(['apt-get', 'update'], check=True)
                subprocess.run(['apt-get', 'install', '-y', 'ffmpeg'], check=True)
                return True
            elif sys.platform == 'darwin':
                subprocess.run(['brew', 'install', 'ffmpeg'], check=True)
                return True
            else:
                st.error("Automatic FFmpeg installation not supported on this platform")
                return False
        except subprocess.CalledProcessError as e:
            st.error(f"Failed to install FFmpeg: {e}")
            return False

# Progress bar function
def progress_bar(amount_of_time: int) -> None:
    progress_text = "Please wait, Generative models hard at work"
    my_bar = st.progress(0, text=progress_text)
    for percent_complete in range(amount_of_time):
        time.sleep(0.04)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(1)
    my_bar.empty()

# Generate image description using Gemini
def generate_detailed_description(image_path: str) -> str:
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    with open(image_path, 'rb') as img_file:
        image_data = {'mime_type': 'image/jpeg', 'data': img_file.read()}
    
    response = model.generate_content(["Describe this image in detail.", image_data])
    return response.text if response and response.text else "No description generated."

# Generate speech from text
def generate_speech(text: str, filename: str) -> str:
    tts = gTTS(text=text, lang='en')
    audio_file = f"{filename}.mp3"
    tts.save(audio_file)
    return audio_file

# Capture image using Streamlit's native camera
def capture_image_via_streamlit() -> str:
    img_file = st.camera_input("Take a picture")
    if img_file:
        img_path = "captured_image.jpg"
        with open(img_path, "wb") as f:
            f.write(img_file.getbuffer())
        return img_path
    return None

# Convert audio to WAV format with FFmpeg
def convert_to_wav(input_audio_path: str, output_audio_path: str) -> str:
    try:
        AudioSegment.converter = "ffmpeg"
        audio = AudioSegment.from_file(input_audio_path)
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export(output_audio_path, format="wav")
        return output_audio_path
    except Exception as e:
        st.error(f"Audio conversion failed: {str(e)}")
        return None

# Convert speech to text
def speech_to_text(audio_path: str) -> str:
    recognizer = sr.Recognizer()
    wav_path = convert_to_wav(audio_path, "converted_query.wav")
    
    if not wav_path:
        return "Audio conversion failed"

    try:
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            return recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        return "Sorry, I couldn't understand the audio."
    except sr.RequestError:
        return "Speech recognition service is unavailable."

# Generate chat response
def chat_about_image(user_query: str, image_description: str) -> str:
    chat_model = genai.GenerativeModel('gemini-1.5-flash-002')
    prompt = f"""
    The user has captured an image and received the following description:
    {image_description}

    The user asks: {user_query}

    Provide a relevant and accurate response.
    """
    response = chat_model.generate_content(prompt)
    return response.text if response and response.text else "I couldn't generate a response."

# Main app function
def main() -> None:
    st.set_page_config(page_title="Enhanced Image-to-Text Converter", page_icon="🖼️")
    st.markdown(css_code, unsafe_allow_html=True)

    # Check for FFmpeg
    if not ensure_ffmpeg_installed():
        st.error("FFmpeg is required for audio processing. Deployment may need configuration.")
        return

    # Initialize session state
    if 'stage' not in st.session_state:
        st.session_state.stage = 'capture'
    if 'history' not in st.session_state:
        st.session_state.history = []

    # Sidebar content
    with st.sidebar:
        st.write("---")
        st.write("AI App created by @Shaik Sayeed")
        if 'image_path' in st.session_state:
            st.image(st.session_state.image_path, caption="Captured Image", use_container_width=True)

    st.header("Enhanced Real-Time Image-to-Text Converter")
    st.write("Capture an image, listen to a detailed description, and chat about it!")

    # Stage 1: Capture Image
    if st.session_state.stage == 'capture':
        captured_image = capture_image_via_streamlit()
        if captured_image:
            st.session_state.image_path = captured_image
            st.session_state.stage = 'describe'
            st.rerun()

    # Stage 2: Generate Description
    elif st.session_state.stage == 'describe':
        st.image(st.session_state.image_path, caption="Captured Image", use_container_width=True)
        progress_bar(100)
        description = generate_detailed_description(st.session_state.image_path)
        st.session_state.description = description
        description_audio = generate_speech(description, "description")
        st.audio(description_audio)
        with st.expander("Generated Image Description"):
            st.write(description)
        st.session_state.history = [f"Assistant: {description}"]
        if st.button("Proceed to Chat"):
            st.session_state.stage = 'chat'
            st.rerun()

    # Stage 3: Chat with Model
    elif st.session_state.stage == 'chat':
        st.image(st.session_state.image_path, caption="Captured Image", use_container_width=True)
        st.write("Ask questions about the image using your voice!")

        # Display conversation history
        with st.expander("Conversation History"):
            for message in st.session_state.history:
                st.write(message)

        # Audio input for questions
        audio = st_audiorec()
        if audio:
            with open("query.mp3", "wb") as f:
                f.write(audio)

            user_query = speech_to_text("query.mp3")
            if user_query:
                st.write(f"You asked: {user_query}")

                # Generate response
                answer = chat_about_image(user_query, st.session_state.description)
                st.write(f"Assistant: {answer}")

                # Generate and play audio response
                answer_audio = generate_speech(answer, "answer")
                st.audio(answer_audio)

                # Update conversation history
                st.session_state.history.append(f"User: {user_query}")
                st.session_state.history.append(f"Assistant: {answer}")

        # Option to restart
        if st.button("Capture a New Image"):
            st.session_state.stage = 'capture'
            st.session_state.history = []
            st.rerun()

if __name__ == "__main__":
    main()
