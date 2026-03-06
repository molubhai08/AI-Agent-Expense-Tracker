import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import (
    ListSQLDatabaseTool,
    InfoSQLDatabaseTool,
    QuerySQLDatabaseTool,
)
from langchain.tools import tool
from crewai import Agent, Crew, Task
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import json
import time
import pyaudio
import wave
from vosk import Model, KaldiRecognizer
from gtts import gTTS
from sarvamai import SarvamAI
import tempfile
import base64

# Page config
st.set_page_config(
    page_title="🏦 Expense Tracker",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        font-weight: bold;
    }
    .recording-box {
        background-color: #ffe6e6;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #ff4444;
        text-align: center;
        font-size: 1.2rem;
    }
    .success-box {
        background-color: #000000;
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #44ff44;
        margin: 10px 0;
    }
    .info-box {
        background-color: #000000;
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #4488ff;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'processing' not in st.session_state:
    st.session_state.processing = False

# Load environment variables
load_dotenv()

# Configuration
@st.cache_resource
def initialize_services():
    """Initialize all services"""
    os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
    SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
    
    # Initialize Sarvam AI
    sarvam_client = SarvamAI(api_subscription_key=SARVAM_API_KEY)
    
    # Load Vosk Hindi model
    VOSK_MODEL_PATH = r"D:\audio_models\vosk-model-hi-0.22\vosk-model-hi-0.22"
    vosk_model = Model(VOSK_MODEL_PATH)
    
    # Database setup
    MYSQL_USER = "root"
    MYSQL_PASSWORD = "naruto"
    MYSQL_HOST = "localhost"
    MYSQL_DB = "expense_tracker"
    MYSQL_PORT = 3306
    
    db = SQLDatabase.from_uri(
        f"mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
    )
    
    return sarvam_client, vosk_model, db

try:
    sarvam_client, vosk_model, db = initialize_services()
except Exception as e:
    st.error(f"Failed to initialize services: {e}")
    st.stop()

# Tools
@tool("list_accounts")
def list_accounts() -> str:
    """List all expense accounts (tables)"""
    return ListSQLDatabaseTool(db=db).invoke("")

@tool("account_details")
def account_details(account_name: str) -> str:
    """Get schema and sample rows for an account. Input: account_name"""
    return InfoSQLDatabaseTool(db=db).invoke(account_name)

@tool("run_query")
def run_query(sql: str) -> str:
    """Execute SQL query. Input: valid SQL statement"""
    return QuerySQLDatabaseTool(db=db).invoke(sql)

@tool("get_date")
def get_date(offset_days: int = 0) -> str:
    """Get date with offset. Input: 0 for today, -7 for week ago, etc."""
    date = datetime.now() + timedelta(days=offset_days)
    return date.strftime("%Y-%m-%d")

@tool("create_account")
def create_account(name: str) -> str:
    """Create expense account. Input: person_name"""
    query = f"""CREATE TABLE IF NOT EXISTS {name} (
        id INT AUTO_INCREMENT PRIMARY KEY,
        date DATE NOT NULL,
        amount DECIMAL(10,2) NOT NULL,
        description VARCHAR(255),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )"""
    try:
        QuerySQLDatabaseTool(db=db).invoke(query)
        return f"✅ Account '{name}' created"
    except Exception as e:
        return f"❌ Error: {str(e)}"

@tool("add_expense")
def add_expense(account: str, amount: float, date: str, desc: str = "") -> str:
    """Add expense. Input: account_name, amount, date (YYYY-MM-DD), description"""
    query = f"INSERT INTO {account} (date, amount, description) VALUES ('{date}', {amount}, '{desc}')"
    try:
        QuerySQLDatabaseTool(db=db).invoke(query)
        return f"✅ Added ${amount} to {account} on {date}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Agent
@st.cache_resource
def get_agent():
    return Agent(
        role="Expense Manager",
        goal="Manage expense accounts: create accounts, log expenses, query data",
        backstory="""Expert at tracking expenses in rupees. Create accounts for people, log their spending with dates, 
        and retrieve expense data. Use get_date for relative dates (today=0, yesterday=-1, week ago=-7).""",
        llm="groq/llama-3.3-70b-versatile",
        tools=[list_accounts, account_details, run_query, get_date, create_account, add_expense],
        allow_delegation=False,
        verbose=True
    )

expense_agent = get_agent()

def record_audio(duration=5):
    """Record audio from microphone"""
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    
    p = pyaudio.PyAudio()
    
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                    input=True, frames_per_buffer=CHUNK)
    
    frames = []
    for _ in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    wf = wave.open(temp_file.name, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    return temp_file.name

def transcribe_hindi(audio_path):
    """Transcribe Hindi audio using Vosk"""
    recognizer = KaldiRecognizer(vosk_model, 16000)
    
    with open(audio_path, "rb") as f:
        while True:
            data = f.read(4000)
            if len(data) == 0:
                break
            recognizer.AcceptWaveform(data)
    
    result = json.loads(recognizer.FinalResult())
    return result.get("text", "")

def translate_to_english(hindi_text):
    """Translate Hindi to English using Sarvam AI"""
    translation = sarvam_client.text.translate(
        input=hindi_text,
        source_language_code="hi-IN",
        target_language_code="en-IN",
        speaker_gender="Male"
    )
    return translation.translated_text

def translate_to_hindi(english_text):
    """Translate English to Hindi using Sarvam AI"""
    translation = sarvam_client.text.translate(
        input=english_text,
        source_language_code="en-IN",
        target_language_code="hi-IN",
        speaker_gender="Male"
    )
    return translation.translated_text


def text_to_speech_hindi(text):
    """Convert Hindi text to speech using Sarvam AI"""
    
    audio_resp = sarvam_client.text_to_speech.convert(
        text=text,
        target_language_code="hi-IN",
        speaker="anushka",      # or other supported voices
        model="bulbul:v2"
    )

    # Create temp WAV file
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)  # important on Windows

    # Decode & write audio bytes
    with open(path, "wb") as f:
        for chunk in audio_resp.audios:
            f.write(base64.b64decode(chunk))

    return path


def process_query(user_input: str) -> str:
    """Process user query and return result"""
    task = Task(
        description=f"Handle: {user_input}",
        expected_output=(
    "Clear confirmation of action taken. "
    "DO NOT mention any dates in the final response. "
    "Dates may be used internally but must NOT appear in the output. "
    "Response must start with 'Namaste' and end with 'Dhanyawad'."
),
        agent=expense_agent
    )
    
    crew = Crew(
        agents=[expense_agent],
        tasks=[task],
        verbose=False
    )
    
    result = crew.kickoff()
    return str(result)

def autoplay_audio(file_path: str):
    """Autoplay audio in Streamlit"""
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(md, unsafe_allow_html=True)

# Main UI
st.markdown('<div class="main-header">🏦 Expense Tracker</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    mode = st.radio(
        "Input Mode",
        ["💬 Text", "🎤 Voice (Hindi)"],
        index=0
    )
    
    if mode == "🎤 Voice (Hindi)":
        duration = st.slider("Recording Duration (seconds)", 3, 10, 5)
    
    st.divider()
    
    st.header("📊 Quick Actions")
    if st.button("📋 List All Accounts"):
        with st.spinner("Fetching accounts..."):
            result = process_query("List all expense accounts")
            st.success(result)
    
    if st.button("🔄 Clear History"):
        st.session_state.history = []
        st.rerun()
    
    st.divider()
    st.caption("💡 Examples:")
    st.caption("• Create account for John")
    st.caption("• Add $50 to John for lunch today")
    st.caption("• Show John's expenses this week")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Interaction")
    
    if mode == "💬 Text":
        # Text input mode
        with st.form(key="text_form", clear_on_submit=True):
            user_input = st.text_area(
                "Enter your query:",
                placeholder="e.g., Create account for Alice or Add $25 to Alice for groceries",
                height=100
            )
            submit = st.form_submit_button("🚀 Submit", use_container_width=True)
        
        if submit and user_input.strip():
            with st.spinner("Processing..."):
                start_time = time.time()
                
                try:
                    result = process_query(user_input)
                    elapsed = time.time() - start_time
                    
                    # Add to history
                    st.session_state.history.insert(0, {
                        'type': 'text',
                        'input': user_input,
                        'output': result,
                        'time': elapsed,
                        'timestamp': datetime.now().strftime("%H:%M:%S")
                    })
                    
                    st.success("✅ Query processed successfully!")
                    st.markdown(f'<div class="success-box"><strong>Result:</strong> {result}</div>', 
                              unsafe_allow_html=True)
                    st.caption(f"⏱️ Processed in {elapsed:.2f}s")
                    
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    
    else:
        # Voice input mode
        st.info("🎤 Click the button below and speak in Hindi")
        
        if st.button("🔴 Start Recording", use_container_width=True, type="primary"):
            st.session_state.processing = True
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Recording
                status_text.markdown('<div class="recording-box">🎤 Recording... Speak now!</div>', 
                                   unsafe_allow_html=True)
                progress_bar.progress(10)
                
                record_start = time.time()
                audio_file = record_audio(duration=duration)
                record_time = time.time() - record_start
                
                # Transcription
                status_text.info("🔄 Transcribing Hindi audio...")
                progress_bar.progress(30)
                
                transcribe_start = time.time()
                hindi_text = transcribe_hindi(audio_file)
                transcribe_time = time.time() - transcribe_start
                
                if not hindi_text.strip():
                    st.warning("❌ No speech detected. Please try again.")
                    st.session_state.processing = False
                    st.stop()
                
                st.markdown(f'<div class="info-box">🎤 <strong>Hindi:</strong> {hindi_text}</div>', 
                          unsafe_allow_html=True)
                
                # Translation to English
                status_text.info("🔄 Translating to English...")
                progress_bar.progress(50)
                
                translate_start = time.time()
                english_text = translate_to_english(hindi_text)
                translate_time = time.time() - translate_start
                
                st.markdown(f'<div class="info-box">🔤 <strong>English:</strong> {english_text}</div>', 
                          unsafe_allow_html=True)
                
                # Process query
                status_text.info("🤖 Processing with AI agent...")
                progress_bar.progress(70)
                
                agent_start = time.time()
                result = process_query(english_text)
                agent_time = time.time() - agent_start
                
                st.markdown(f'<div class="success-box">🤖 <strong>Result:</strong> {result}</div>', 
                          unsafe_allow_html=True)
                
                # Translate back to Hindi
                status_text.info("🔄 Translating response to Hindi...")
                progress_bar.progress(85)
                
                back_translate_start = time.time()
                hindi_result = translate_to_hindi(result)
                back_translate_time = time.time() - back_translate_start
                
                st.markdown(f'<div class="info-box">🔤 <strong>Hindi Response:</strong> {hindi_result}</div>', 
                          unsafe_allow_html=True)
                
                # Text to speech
                status_text.info("🔊 Generating audio response...")
                progress_bar.progress(95)
                
                tts_start = time.time()
                audio_output = text_to_speech_hindi(hindi_result)
                tts_time = time.time() - tts_start
                
                progress_bar.progress(100)
                status_text.success("✅ Complete! Playing audio...")
                
                # Play audio
                autoplay_audio(audio_output)
                
                # Show timing breakdown
                total_time = (record_time + transcribe_time + translate_time + 
                            agent_time + back_translate_time + tts_time)
                
                with st.expander("⏱️ Timing Breakdown"):
                    st.write(f"🎤 Recording: {record_time:.2f}s")
                    st.write(f"📝 Transcription: {transcribe_time:.2f}s")
                    st.write(f"🔤 Translation: {translate_time:.2f}s")
                    st.write(f"🤖 AI Processing: {agent_time:.2f}s")
                    st.write(f"🔤 Back Translation: {back_translate_time:.2f}s")
                    st.write(f"🔊 Text-to-Speech: {tts_time:.2f}s")
                    st.write(f"**⏱️ Total: {total_time:.2f}s**")
                
                # Add to history
                st.session_state.history.insert(0, {
                    'type': 'voice',
                    'hindi_input': hindi_text,
                    'english_input': english_text,
                    'output': result,
                    'hindi_output': hindi_result,
                    'time': total_time,
                    'timestamp': datetime.now().strftime("%H:%M:%S")
                })
                
                # Cleanup
                os.unlink(audio_file)
                os.unlink(audio_output)
                
            except Exception as e:
                st.error(f"❌ Error during voice processing: {e}")
            
            finally:
                st.session_state.processing = False

with col2:
    st.header("📜 History")
    
    if st.session_state.history:
        for i, entry in enumerate(st.session_state.history[:10]):  # Show last 10
            with st.expander(f"#{i+1} - {entry['timestamp']} ({entry['type'].upper()})"):
                if entry['type'] == 'text':
                    st.write(f"**Input:** {entry['input']}")
                    st.write(f"**Output:** {entry['output']}")
                else:
                    st.write(f"**Hindi Input:** {entry['hindi_input']}")
                    st.write(f"**English Input:** {entry['english_input']}")
                    st.write(f"**Output:** {entry['output']}")
                    st.write(f"**Hindi Output:** {entry['hindi_output']}")
                
                st.caption(f"⏱️ {entry['time']:.2f}s")
    else:
        st.info("No interactions yet. Start by entering a query!")

# Footer
st.divider()
st.caption("🏦 Expense Tracker - Powered by CrewAI, Groq, and Sarvam AI")