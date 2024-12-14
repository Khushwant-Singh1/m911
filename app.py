import streamlit as st
import speech_recognition as sr
import pyttsx3
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import openai
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
import av
import numpy as np
import wave
import os

# Simple CSS for styling user and bot messages
css = """
<style>
.user-message, .bot-message {
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 10px;
    line-height: 1.5;
}
.user-message {
    background:rgb(0, 0, 0); /* black */
    color: white;
}
.bot-message {
    background:rgb(0, 0, 0); /* black */
    color: white;
}
</style>
"""

# HTML templates for user and bot messages
user_template = """
<div class="user-message">
    <strong>User:</strong> {{MSG}}
</div>
"""

bot_template = """
<div class="bot-message">
    <strong>AI:</strong> {{MSG}}
</div>
"""

def text_to_speech(text):
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        st.error(f"⚠ Error in text-to-speech: {e}")

def get_pdf_text(pdf_paths):
    text = ""
    try:
        for pdf_path in pdf_paths:
            pdf_reader = PdfReader(pdf_path)
            for page in pdf_reader.pages:
                text += page.extract_text()
    except Exception as e:
        st.error(f"⚠ Error reading the data: {e}")
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    openai_api_key = st.secrets["openai"]["api_key"]
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def handle_userinput(user_question):
    if 'conversation' not in st.session_state or st.session_state.conversation is None:
        st.error("⚠ Please process the data first.")
        return
    response = st.session_state.conversation({'question': user_question})
    answer = response['answer']
    st.session_state.conversation_history.append((user_question, answer))

    # Display conversation
    for i, (q, a) in enumerate(st.session_state.conversation_history):
        st.write(user_template.replace("{{MSG}}", q), unsafe_allow_html=True)
        st.write(bot_template.replace("{{MSG}}", a), unsafe_allow_html=True)

    text_to_speech(answer)

def transcribe_audio_file(audio_path):
    r = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = r.record(source)
    try:
        text = r.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        st.error("😔 Could not understand the audio.")
    except sr.RequestError:
        st.error("⚠ Could not request results from speech recognition service.")
    return ""

def audio_frame_callback(audio_frame: av.AudioFrame):
    if st.session_state.recording:
        st.session_state.audio_processor.add_frame(audio_frame)
    return audio_frame

class AudioProcessor:
    def __init__(self):
        self.frames = []
        self.sampling_rate = 16000  # default sampling rate (will adjust if necessary)
        self.channels = 1

    def add_frame(self, frame: av.AudioFrame):
        # Convert the audio frame to a numpy array
        # frame.to_ndarray() returns shape: (channels, samples)
        arr = frame.to_ndarray()
        # Assuming mono audio or taking just the first channel if stereo
        if arr.ndim == 2 and arr.shape[0] > 1:
            arr = arr[0, :]  # take first channel
        elif arr.ndim == 2 and arr.shape[0] == 1:
            arr = arr[0, :]
        # Convert to int16 for WAV
        arr = arr.astype(np.int16)
        self.frames.append(arr)

    def save_wav(self, filename):
        # Combine all frames into one numpy array
        if not self.frames:
            return
        audio_data = np.concatenate(self.frames, axis=0)
        # Write to a WAV file
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(2)  # 16 bits
        wf.setframerate(self.sampling_rate)
        wf.writeframes(audio_data.tobytes())
        wf.close()



def main():
    st.set_page_config(page_title="Understand IT by Metaverse911", page_icon="💻")

    SYSTEM_MESSAGE = st.secrets.get("openai", {}).get("SYSTEM_MESSAGE", "You are a helpful AI assistant.")

    st.markdown(css, unsafe_allow_html=True)

    # Session states
    if "audio_processor" not in st.session_state:
        st.session_state.audio_processor = AudioProcessor()
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "audio_processor" not in st.session_state:
        st.session_state.audio_processor = AudioProcessor()
    if "recording" not in st.session_state:
        st.session_state.recording = False

    # Main header
    st.markdown("## 📚 Understand IT by Metaverse911")
    st.markdown("Ask questions about your documents, either by typing or by speaking! 🎤💬")

    input_method = st.radio("Choose input method:", ["Text Input 💻", "Voice Input 🎤"], index=0)

    # If Text Input
    if input_method == "Text Input 💻":
        user_question = st.text_input("🤔 Ask a question about your documents:")
        if user_question:
            handle_userinput(user_question)
    else:
        # Voice Input Method
        st.write("*Voice Input Instructions:*")
        st.write("1. Click 'Start Recording' and allow microphone access in your browser.")
        st.write("2. Speak your question.")
        st.write("3. Click 'Stop and Transcribe' when done.")

        # Buttons to control recording
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎙 Start Recording"):
                st.session_state.recording = True
                st.session_state.audio_processor.frames = []
        with col2:
            if st.button("🛑 Stop and Transcribe"):
                st.session_state.recording = False
                # Save recorded audio
                audio_filename = "temp_audio.wav"
                st.session_state.audio_processor.save_wav(audio_filename)

                if os.path.exists(audio_filename):
                    voice_question = transcribe_audio_file(audio_filename)
                    os.remove(audio_filename)
                    if voice_question:
                        st.write(f"*Recognized:* {voice_question}")
                        handle_userinput(voice_question)

        # WebRTC audio capture
 
        webrtc_streamer(
            key="speech",
            mode=WebRtcMode.SENDRECV,
            client_settings=ClientSettings(
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                media_stream_constraints={"audio": True, "video": False},
            ),
            # Use the audio frame callback
        )


    with st.sidebar:
        st.markdown("### 🔧 Process the Data")
        st.markdown("Click below to process the data before asking questions. ")

        # Hardcoded Data paths
        pdf_path_1 = "Ilesh Sir (IK) - Words.pdf"  # Replace with your PDF path
        pdf_path_2 = "UBIK SOLUTION.pdf"  # Replace with your second PDF path
        pdf_paths = [pdf_path_1, pdf_path_2]

        if st.button("🚀 Process Data"):
            with st.spinner("🔄 Processing Data..."):
                raw_text = get_pdf_text(pdf_paths)
                if raw_text.strip():
                    text_chunks = get_text_chunks(raw_text)
                    st.session_state.vectorstore = get_vectorstore(text_chunks)
                    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, openai_api_key = st.secrets["openai"]["api_key"])
                    st.session_state.conversation = ConversationalRetrievalChain.from_llm(
                        llm=llm,
                        retriever=st.session_state.vectorstore.as_retriever(),
                        memory=st.session_state.memory
                    )
                    st.success("✅ DATA processed successfully!")
                else:
                    st.error("⚠ No text could be extracted from the PDFs.")
        
        st.markdown("---")

        if st.button("🧹 Clear Chat History"):
            st.session_state.memory.clear()
            st.session_state.conversation_history = []
            if st.session_state.vectorstore is not None:
                llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, openai_api_key = st.secrets["openai"]["api_key"])
                st.session_state.conversation = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=st.session_state.vectorstore.as_retriever(),
                    memory=st.session_state.memory
                )
            st.experimental_rerun()

if __name__ == '__main__':
    main()