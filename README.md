# 🏦 Multilingual Voice AI Expense Tracker

An end-to-end voice-enabled expense management system that integrates Speech-to-Text (STT), Large Language Models (LLM), and Text-to-Speech (TTS) to create a seamless Hindi voice interface for expense tracking.

## 🎯 Project Overview

This project demonstrates a production-ready AI workflow that processes Hindi voice commands, translates them to English, executes database operations using an AI agent, and responds back in Hindi with voice output. Built to showcase practical AI deployment skills including prompt engineering, multi-modal AI integration, and agentic workflows.

## ✨ Key Features

- **🎤 Hindi Voice Input**: Real-time audio recording and transcription using Vosk
- **🔄 Bidirectional Translation**: Hindi ↔ English translation via Sarvam AI
- **🤖 AI Agent**: CrewAI-powered agent for natural language SQL operations
- **🔊 Voice Response**: Text-to-speech output in Hindi
- **💾 Database Management**: MySQL integration for expense tracking
- **📊 Interactive UI**: Streamlit-based interface with real-time feedback
- **⏱️ Performance Monitoring**: Detailed timing breakdown for each pipeline stage

## 🛠️ Tech Stack

### AI/ML Components
- **LLM**: Groq (Llama 3.3 70B)
- **STT**: Vosk (Hindi model)
- **TTS**: Sarvam AI + gTTS
- **Translation**: Sarvam AI
- **Agent Framework**: CrewAI
- **LLM Orchestration**: LangChain

### Backend & Database
- **Database**: MySQL
- **ORM**: SQLAlchemy
- **API Integration**: Groq API, Sarvam AI API

### Frontend
- **UI Framework**: Streamlit
- **Audio Processing**: PyAudio, Wave

## 📋 Prerequisites

- Python 3.8+
- MySQL Server
- Microphone for voice input
- API Keys:
  - Groq API Key
  - Sarvam AI API Key

## 🚀 Installation

1. Clone the repository
```bash
git clone <repository-url>
cd expense-tracker
```

2. Install dependencies
```bash
pip install streamlit langchain langchain-community crewai groq python-dotenv
pip install mysql-connector-python sqlalchemy pyaudio wave vosk gtts sarvamai
```

3. Download Vosk Hindi Model
```bash
# Download from: https://alphacephei.com/vosk/models
# Extract to: D:\audio_models\vosk-model-hi-0.22\
```

4. Set up MySQL Database
```sql
CREATE DATABASE expense_tracker;
```

5. Configure Environment Variables
Create a `.env` file:
```env
GROQ_API_KEY=your_groq_api_key
SARVAM_API_KEY=your_sarvam_api_key
```

6. Update database credentials in the code
```python
MYSQL_USER = "your_username"
MYSQL_PASSWORD = "your_password"
MYSQL_HOST = "localhost"
MYSQL_DB = "expense_tracker"
```

## 💻 Usage

### Running the Application

```bash
streamlit run final2.py
```

### Voice Mode (Hindi)
1. Select "🎤 Voice (Hindi)" mode
2. Click "🔴 Start Recording"
3. Speak your command in Hindi
4. System will:
   - Transcribe your speech
   - Translate to English
   - Process with AI agent
   - Translate response to Hindi
   - Play audio response

### Text Mode
1. Select "💬 Text" mode
2. Enter commands in English
3. Get instant text responses

### Example Commands

**Hindi Voice:**
- "राज के लिए खाता बनाओ" (Create account for Raj)
- "राज ने आज पचास रुपये खर्च किए" (Raj spent 50 rupees today)
- "राज के खर्चे दिखाओ" (Show Raj's expenses)

**English Text:**
- "Create account for John"
- "Add $50 to John for lunch today"
- "Show John's expenses this week"

## 🏗️ Architecture

```
User Voice Input (Hindi)
    ↓
[STT] Vosk Transcription
    ↓
[Translation] Hindi → English (Sarvam AI)
    ↓
[LLM Agent] CrewAI + Groq (SQL Operations)
    ↓
[Translation] English → Hindi (Sarvam AI)
    ↓
[TTS] Sarvam AI Voice Synthesis
    ↓
Audio Output (Hindi)
```

## 📊 Performance Metrics

Typical pipeline latency breakdown:
- 🎤 Recording: ~5s (configurable)
- 📝 Transcription: ~1-2s
- 🔤 Translation: ~0.5-1s
- 🤖 AI Processing: ~2-3s
- 🔤 Back Translation: ~0.5-1s
- 🔊 Text-to-Speech: ~1-2s

**Total End-to-End Latency**: ~10-15 seconds

## 🎯 AI Agent Capabilities

The CrewAI agent can:
- ✅ Create expense accounts (database tables)
- ✅ Add expenses with dates and descriptions
- ✅ Query expenses by date range
- ✅ Handle relative dates ("today", "yesterday", "last week")
- ✅ Execute complex SQL queries
- ✅ Provide natural language responses

## 🔧 Project Structure

```
.
├── final2.py           # Main Streamlit application
├── new.py              # CLI version
├── test.py             # Audio transcription test
├── .env                # Environment variables
├── README.md           # Documentation
└── requirements.txt    # Dependencies (to be created)
```

## 🚧 Known Limitations

- Requires local Vosk model download (~1GB)
- Hindi transcription accuracy depends on audio quality
- Database credentials hardcoded (should use env vars)
- No user authentication
- Single-user session state

## 🔮 Future Enhancements

- [ ] Add evaluation metrics for transcription accuracy
- [ ] Implement A/B testing for different TTS models
- [ ] Add support for multiple languages
- [ ] Build automated testing pipeline
- [ ] Add user authentication
- [ ] Deploy to cloud (AWS/GCP)
- [ ] Add expense analytics and visualization
- [ ] Implement conversation history persistence

## 📝 Skills Demonstrated

- ✅ LLM prompt engineering for production use cases
- ✅ Building AI workflows with STT, TTS, and LLM components
- ✅ API integration and orchestration
- ✅ Agentic system design with CrewAI
- ✅ Real-time audio processing
- ✅ Database operations via natural language
- ✅ Full-stack application development

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

**Sarthak**
- Email: sarthak.molu08@gmail.com
- LinkedIn: [LinkedIn](https://www.linkedin.com/in/sarthak-5272b32b2)
- GitHub: [GitHub](https://github.com/molubhai08)
- Portfolio: [Portfolio](https://molubhai08.github.io/new_portfolio/)

## 🙏 Acknowledgments

- Groq for LLM API
- Sarvam AI for translation and TTS
- Vosk for offline STT
- CrewAI for agent framework
- LangChain for LLM orchestration

---

**Note**: This project was built to demonstrate practical AI engineering skills including multi-modal AI integration, prompt engineering, and production-ready workflow design.
