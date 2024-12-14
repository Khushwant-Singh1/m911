import streamlit as st
import pyttsx3
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import openai

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
    background:rgb(0, 0, 0); /* light green */
}
.bot-message {
    background:rgb(0, 0, 0); /* light gray */
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
        st.error(f"⚠️ Error in text-to-speech: {e}")

def get_pdf_text(pdf_paths):
    text = ""
    try:
        for pdf_path in pdf_paths:
            pdf_reader = PdfReader(pdf_path)
            for page in pdf_reader.pages:
                text += page.extract_text()
    except Exception as e:
        st.error(f"⚠️ Error reading the data: {e}")
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
        st.error("⚠️ Please process the data first.")
        return
    response = st.session_state.conversation({'question': user_question})
    answer = response['answer']
    st.session_state.conversation_history.append((user_question, answer))

    # Display conversation
    for i, (q, a) in enumerate(st.session_state.conversation_history):
        st.write(user_template.replace("{{MSG}}", q), unsafe_allow_html=True)
        st.write(bot_template.replace("{{MSG}}", a), unsafe_allow_html=True)

    text_to_speech(answer)

# JavaScript for capturing speech input
speech_input_js = """
<script>
    const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    recognition.lang = 'en-US';
    recognition.interimResults = true;
    recognition.onstart = () => {
        document.getElementById('status').innerText = 'Listening...';
    };
    recognition.onresult = (event) => {
        let transcript = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
            transcript += event.results[i][0].transcript;
        }
        window.parent.postMessage({type: 'speech_result', text: transcript}, '*');
    };
    recognition.onerror = (event) => {
        document.getElementById('status').innerText = 'Error occurred: ' + event.error;
    };
    function startListening() {
        recognition.start();
    }
</script>
<button onclick="startListening()">Start Voice Input</button>
<p id="status">Click the button to start listening.</p>
</script>
"""

def main():
    st.set_page_config(page_title="Understand IT by Metaverse911", page_icon="💻")

    SYSTEM_MESSAGE = st.secrets.get("openai", {}).get("SYSTEM_MESSAGE", "You are a helpful AI assistant.")

    st.markdown(css, unsafe_allow_html=True)

    # Session states
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    # Main header
    st.markdown("## 📚 Understand IT by Metaverse911")
    st.markdown("Ask questions about your documents, either by typing or by speaking! 🎤💬")

    # Input method selection
    input_method = st.radio("**Choose input method:**", ["Text Input 💻", "Voice Input 🎤"], index=0)

    if input_method == "Text Input 💻":
        user_question = st.text_input("🤔 **Ask a question about your documents:**")
        if user_question:
            handle_userinput(user_question)
    else:
        # Include JavaScript to capture voice input and send it to the Streamlit app
        st.components.v1.html(speech_input_js, height=400)

        # Listen for messages from JavaScript
        message = st.experimental_get_query_params().get('speech_result', None)
        if message:
            voice_question = message[0]
            if voice_question:
                handle_userinput(voice_question)

    with st.sidebar:
        st.markdown("### 🔧 **Process the Data**")
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
                    st.error("⚠️ No text could be extracted from the PDFs.")
        
        st.markdown("---")

        if st.button("🧹 Clear Chat History"):
            st.session_state.memory.clear()
            st.session_state.conversation_history = []
            if st.session_state.vectorstore is not None:
                llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
                st.session_state.conversation = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=st.session_state.vectorstore.as_retriever(),
                    memory=st.session_state.memory
                )
            st.experimental_rerun()

if __name__ == '__main__':
    main()
