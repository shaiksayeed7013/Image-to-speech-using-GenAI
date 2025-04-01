# main.py - Complete Improved Version
import os
import time
import io
import wave
import numpy as np
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from gtts import gTTS
from PIL import Image
import speech_recognition as sr
from st_audiorec import st_audiorec
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
    .stAudioRecorder {
        margin-bottom: 20px;
    }
    </style>
"""

def progress_bar(amount_of_time: int) -> None:
    progress_text = "Processing your request..."
    my_bar = st.progress(0, text=progress_text)
    for percent_complete in range(amount_of_time):
        time.sleep(0.04)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(1)
    my_bar.empty()

def generate_detailed_description(image_path: str) -> str:
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    with open(image_path, 'rb') as img_file:
        image_data = {'mime_type': 'image/jpeg', 'data': img_file.read()}
    
    response = model.generate_content(["Describe this image in detail.", image_data])
    return response.text if response and response.text else "No description generated."

def generate_speech(text: str, filename: str) -> str:
    tts = gTTS(text=text, lang='en', slow=False)
    audio_file = f"{filename}.mp3"
    tts.save(audio_file)
    return audio_file

def capture_image_via_streamlit() -> str:
    st.info("Please position your object in the frame and click the button below")
    img_file = st.camera_input("Take a clear picture of your object")
    if img_file:
        img_path = "captured_image.jpg"
        with open(img_path, "wb") as f:
            f.write(img_file.getbuffer())
        return img_path
    return None

def process_audio(audio_bytes: bytes) -> bytes:
    """Enhance audio quality for better recognition"""
    try:
        # Convert bytes to numpy array
        with wave.open(io.BytesIO(audio_bytes)) as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            audio_array = np.frombuffer(frames, dtype=np.int16)
        
        # Normalize volume and reduce background noise
        max_val = np.max(np.abs(audio_array))
        if max_val > 0:
            audio_array = (audio_array / max_val * 32767).astype(np.int16)
            # Simple noise reduction (threshold filter)
            audio_array[np.abs(audio_array) < 3000] = 0
        
        # Convert back to bytes
        with io.BytesIO() as output:
            with wave.open(output, 'wb') as wav_out:
                wav_out.setnchannels(1)
                wav_out.setsampwidth(2)
                wav_out.setframerate(16000)  # Standard speech recognition rate
                wav_out.writeframes(audio_array.tobytes())
            return output.getvalue()
    except Exception as e:
        st.warning(f"Audio enhancement warning: {e}")
        return audio_bytes  # Return original if processing fails

def speech_to_text(audio_bytes: bytes) -> str:
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 4000  # Better for web recordings
    recognizer.dynamic_energy_threshold = True
    
    try:
        # Process audio first
        processed_audio = process_audio(audio_bytes)
        
        with io.BytesIO(processed_audio) as audio_file:
            with wave.open(audio_file, 'rb') as wav_file:
                audio_data = sr.AudioData(
                    wav_file.readframes(wav_file.getnframes()),
                    sample_rate=wav_file.getframerate(),
                    sample_width=wav_file.getsampwidth()
                )
                
                # Try Google Web Speech API first
                try:
                    text = recognizer.recognize_google(audio_data, language="en-US")
                    if not text.strip():
                        raise ValueError("Empty transcription")
                    return text
                except (sr.UnknownValueError, ValueError):
                    # Fallback to whisper if available
                    try:
                        return recognizer.recognize_whisper(audio_data, language="english", model="base")
                    except:
                        return None
    
    except Exception as e:
        st.error(f"Audio processing error: {str(e)}")
        return None

def chat_about_image(user_query: str, image_description: str) -> str:
    chat_model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""
    The user has captured an image with the following description:
    {image_description}

    The user asked: {user_query}

    Provide a helpful and accurate response focusing on the image content.
    If the question seems unrelated to the image, politely clarify.
    """
    response = chat_model.generate_content(prompt)
    return response.text if response and response.text else "I couldn't generate a response."

def main() -> None:
    st.set_page_config(
        page_title="Image Insight Assistant",
        page_icon="🖼️",
        layout="centered"
    )
    st.markdown(css_code, unsafe_allow_html=True)

    # Initialize session state
    if 'stage' not in st.session_state:
        st.session_state.stage = 'capture'
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'retry_count' not in st.session_state:
        st.session_state.retry_count = 0

    # Sidebar
    with st.sidebar:
        st.title("Image Insight")
        st.write("Capture an image and ask questions about it")
        if 'image_path' in st.session_state:
            st.image(st.session_state.image_path, caption="Your Image", use_container_width=True)
        st.write("---")
        st.write("Created by [Your Name]")

    # Main content
    st.title("Image Insight Assistant")
    st.write("Capture an object, get its description, and ask questions about it!")

    # Stage 1: Capture Image
    if st.session_state.stage == 'capture':
        st.subheader("Step 1: Capture Your Image")
        captured_image = capture_image_via_streamlit()
        if captured_image:
            st.session_state.image_path = captured_image
            st.session_state.stage = 'describe'
            st.rerun()

    # Stage 2: Generate Description
    elif st.session_state.stage == 'describe':
        st.subheader("Step 2: Image Analysis")
        st.image(st.session_state.image_path, caption="Your Captured Image", use_container_width=True)
        
        progress_bar(100)
        
        description = generate_detailed_description(st.session_state.image_path)
        st.session_state.description = description
        
        st.success("Analysis Complete!")
        with st.expander("📝 View Detailed Description"):
            st.write(description)
        
        description_audio = generate_speech(description, "description")
        st.audio(description_audio)
        
        if st.button("Ask Questions About This Image", type="primary"):
            st.session_state.stage = 'chat'
            st.rerun()

    # Stage 3: Chat with Model
    elif st.session_state.stage == 'chat':
        st.subheader("Step 3: Ask About Your Image")
        st.image(st.session_state.image_path, caption="Your Image", use_container_width=True)
        
        # Conversation history
        with st.expander("💬 Conversation History"):
            if not st.session_state.history:
                st.write("No questions asked yet")
            for msg in st.session_state.history:
                st.write(msg)
        
        # Audio input
        st.write("**Ask your question:**")
        audio_bytes = st_audiorec()
        
        if audio_bytes:
            with st.spinner("Processing your question..."):
                user_query = speech_to_text(audio_bytes)
                
                if user_query:
                    st.session_state.retry_count = 0
                    st.session_state.history.append(f"👤 You: {user_query}")
                    
                    answer = chat_about_image(user_query, st.session_state.description)
                    st.session_state.history.append(f"🤖 Assistant: {answer}")
                    
                    st.write(f"**You asked:** {user_query}")
                    st.write(f"**Assistant:** {answer}")
                    
                    answer_audio = generate_speech(answer, "answer")
                    st.audio(answer_audio)
                else:
                    st.session_state.retry_count += 1
                    if st.session_state.retry_count <= 2:
                        st.warning("I didn't catch that. Please try speaking again clearly.")
                    else:
                        st.error("Audio capture failed. Please type your question instead.")
                        manual_query = st.text_input("Or type your question here:")
                        if manual_query:
                            answer = chat_about_image(manual_query, st.session_state.description)
                            st.write(f"**Assistant:** {answer}")
                            answer_audio = generate_speech(answer, "answer")
                            st.audio(answer_audio)

        # Navigation buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Ask Another Question"):
                st.session_state.retry_count = 0
                st.rerun()
        with col2:
            if st.button("📸 Capture New Image"):
                st.session_state.stage = 'capture'
                st.session_state.history = []
                st.session_state.retry_count = 0
                st.rerun()

if __name__ == "__main__":
    main()
