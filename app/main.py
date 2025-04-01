# main.py - Final Version with Persistent Description
import os
import time
import io
import base64
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
    .recording-instructions {
        font-size: 0.9em;
        color: #666;
        margin-top: -15px;
        margin-bottom: 15px;
    }
    .image-description {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    </style>
"""

def progress_bar(amount_of_time: int) -> None:
    progress_text = "Analyzing your image..."
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
    
    response = model.generate_content([
        "Describe this image in comprehensive detail, including:",
        "1. Main objects and their attributes (colors, shapes, positions)",
        "2. Background elements",
        "3. Any text or recognizable logos",
        "4. Overall style and atmosphere",
        image_data
    ])
    return response.text if response and response.text else "No description generated."

def generate_speech(text: str, filename: str) -> str:
    tts = gTTS(text=text, lang='en', slow=False)
    audio_file = f"{filename}.mp3"
    tts.save(audio_file)
    return audio_file

def capture_image_via_streamlit() -> str:
    st.info("Position your object clearly in frame")
    img_file = st.camera_input("Take a picture (click the button below)")
    if img_file:
        img_path = "captured_image.jpg"
        with open(img_path, "wb") as f:
            f.write(img_file.getbuffer())
        return img_path
    return None

def process_audio_data(audio_bytes: bytes) -> bytes:
    """Convert audio to proper format for recognition"""
    try:
        # Convert the recorded bytes to base64
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        return audio_bytes  # Already in correct format
    except Exception as e:
        st.warning(f"Audio processing note: {e}")
        return audio_bytes

def speech_to_text(audio_bytes: bytes) -> str:
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 3000
    recognizer.dynamic_energy_threshold = True
    
    try:
        processed_audio = process_audio_data(audio_bytes)
        
        with io.BytesIO(processed_audio) as audio_file:
            audio_data = sr.AudioFile(audio_file)
            
            with audio_data as source:
                audio = recognizer.record(source)
                try:
                    text = recognizer.recognize_google(audio, language="en-US")
                    if not text.strip():
                        raise ValueError("Empty transcription")
                    return text
                except (sr.UnknownValueError, ValueError) as e:
                    st.warning("Couldn't understand audio. Please try speaking more clearly.")
                    return None
    
    except Exception as e:
        st.error(f"Audio processing error: {str(e)}")
        return None

def chat_about_image(user_query: str, image_description: str) -> str:
    chat_model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""
    IMAGE DESCRIPTION:
    {image_description}

    USER QUESTION: {user_query}

    Provide a detailed, helpful response focusing on the image content.
    If the question seems unrelated, politely point this out while still trying to help.
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
    if 'description' not in st.session_state:
        st.session_state.description = None

    # Sidebar
    with st.sidebar:
        st.title("Image Insight")
        st.write("Capture ➔ Describe ➔ Chat")
        if 'image_path' in st.session_state:
            st.image(st.session_state.image_path, caption="Your Image", use_container_width=True)
        st.write("---")
        st.write("Created by [Your Name]")

    # Main content
    st.title("Image Insight Assistant")
    
    # Always show description if available
    if st.session_state.description:
        st.markdown('<div class="image-description">📝 <strong>Image Description:</strong></div>', unsafe_allow_html=True)
        st.write(st.session_state.description)
        st.audio(generate_speech(st.session_state.description, "description"))

    # Stage 1: Capture Image
    if st.session_state.stage == 'capture':
        st.subheader("Step 1: Capture Your Image")
        captured_image = capture_image_via_streamlit()
        if captured_image:
            st.session_state.image_path = captured_image
            st.session_state.stage = 'describe'
            st.session_state.history = []
            st.session_state.retry_count = 0
            st.rerun()

    # Stage 2: Generate Description
    elif st.session_state.stage == 'describe':
        st.subheader("Step 2: Generating Description...")
        progress_bar(100)
        
        description = generate_detailed_description(st.session_state.image_path)
        st.session_state.description = description
        st.session_state.stage = 'chat'
        st.rerun()

    # Stage 3: Chat with Model
    elif st.session_state.stage == 'chat':
        st.subheader("Step 3: Ask About Your Image")
        
        # Conversation history
        with st.expander("💬 Conversation History"):
            if not st.session_state.history:
                st.write("No questions asked yet")
            for msg in st.session_state.history:
                st.write(msg)
        
        # Audio input with better instructions
        st.write("**Ask your question:**")
        st.markdown('<div class="recording-instructions">Press record, wait 1 second, then speak clearly</div>', unsafe_allow_html=True)
        
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
                        st.error("Audio capture failed. Please type your question:")
                        manual_query = st.text_input("Type your question here:")
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
                st.session_state.description = None
                st.session_state.history = []
                st.session_state.retry_count = 0
                st.rerun()

if __name__ == "__main__":
    main()
