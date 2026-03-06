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
import subprocess
import platform

load_dotenv()

# Set API keys
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
expense_agent = Agent(
    role="Expense Manager",
    goal="Manage expense accounts: create accounts, log expenses, query data",
    backstory="""Expert at tracking expenses. Create accounts for people, log their spending with dates, 
    and retrieve expense data. Use get_date for relative dates (today=0, yesterday=-1, week ago=-7).""",
    llm="groq/llama-3.3-70b-versatile",
    tools=[list_accounts, account_details, run_query, get_date, create_account, add_expense],
    allow_delegation=False,
    verbose=True
)


def record_audio(duration=5, filename="temp_recording.wav"):
    """Record audio from microphone"""
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    
    p = pyaudio.PyAudio()
    
    print(f"🎤 Recording for {duration} seconds...")
    
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                    input=True, frames_per_buffer=CHUNK)
    
    frames = []
    for _ in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)
    
    print("✅ Recording finished")
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    return filename


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


def text_to_speech_hindi(text, output_file="response.mp3"):
    """Convert Hindi text to speech"""
    tts = gTTS(text=text, lang='hi')
    tts.save(output_file)
    return output_file


def play_audio(audio_path):
    """Play audio file based on OS"""
    system = platform.system()
    try:
        if system == "Windows":
            os.startfile(audio_path)
        elif system == "Darwin":  # macOS
            subprocess.call(["open", audio_path])
        else:  # Linux
            subprocess.call(["xdg-open", audio_path])
    except Exception as e:
        print(f"❌ Could not play audio: {e}")


def process_query(user_input: str) -> str:
    """Process user query and return result"""
    task = Task(
        description=f"Handle: {user_input}",
        expected_output="Clear confirmation of action taken with relevant data",
        agent=expense_agent
    )
    
    crew = Crew(
        agents=[expense_agent],
        tasks=[task],
        verbose=False
    )
    
    result = crew.kickoff()
    return str(result)


def voice_interaction():
    """Handle voice input and output"""
    total_start = time.time()



    
    # Record audio
    record_start = time.time()
    audio_file = record_audio(duration=5)
    record_time = time.time() - record_start
    print(f"⏱️  Recording time: {record_time:.2f}s")
    
    # Transcribe Hindi
    transcribe_start = time.time()
    hindi_text = transcribe_hindi(audio_file)
    transcribe_time = time.time() - transcribe_start
    print(f"🎤 Hindi Text: {hindi_text}")
    print(f"⏱️  Transcription time: {transcribe_time:.2f}s")


    
    if not hindi_text.strip():
        print("❌ No speech detected")
        return
    
    # Translate to English
    translate_start = time.time()
    english_text = translate_to_english(hindi_text)
    translate_time = time.time() - translate_start
    print(f"🔤 English: {english_text}")
    print(f"⏱️  Translation time: {translate_time:.2f}s")


    
    # Process with agent
    agent_start = time.time()
    result = process_query(english_text)
    agent_time = time.time() - agent_start
    print(f"\n🤖 Agent Result: {result}")
    print(f"⏱️  Agent processing time: {agent_time:.2f}s")
    
    # Translate result to Hindi
    back_translate_start = time.time()
    hindi_result = translate_to_hindi(result)
    back_translate_time = time.time() - back_translate_start
    print(f"🔤 Hindi Response: {hindi_result}")
    print(f"⏱️  Back translation time: {back_translate_time:.2f}s")

    
    # Text to speech
    tts_start = time.time()
    audio_output = text_to_speech_hindi(hindi_result)
    tts_time = time.time() - tts_start
    print(f"⏱️  TTS time: {tts_time:.2f}s")
    
    # Play audio
    print("🔊 Playing response...")
    play_audio(audio_output)
    
    total_time = time.time() - total_start
    print(f"\n⏱️  TOTAL TIME: {total_time:.2f}s\n")


# Main interface
if __name__ == "__main__":
    print("🏦 Expense Tracker Started")
    print("\nModes:")
    print("1. Type 'voice' - Voice input (Hindi)")
    print("2. Type text directly - Text input (English)")
    print("3. Type 'quit' - Exit\n")
    
    while True:
        mode = input("Choose mode (voice/text/quit): ").strip().lower()
        
        if mode in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if mode == 'voice':
            try:
                voice_interaction()
            except Exception as e:
                print(f"❌ Voice error: {e}\n")
        
        else:
            # Text mode
            query = input("\nYou: ").strip()
            
            if not query:
                continue
            
            try:
                start = time.time()
                result = process_query(query)
                elapsed = time.time() - start
                print(f"\n🤖 Assistant: {result}")
                print(f"⏱️  Processing time: {elapsed:.2f}s\n")
            except Exception as e:
                print(f"❌ Error: {e}\n")