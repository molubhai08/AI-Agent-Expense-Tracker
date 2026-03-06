from vosk import Model, KaldiRecognizer
from pydub import AudioSegment
from gtts import gTTS
import json
import os
from dotenv import load_dotenv
import requests
import time

start = time.time()

# --------------------------
# 1. Load the Vosk Hindi Model
# --------------------------
model = Model(r"D:\audio_models\vosk-model-small-hi-0.22\vosk-model-small-hi-0.22")

# --------------------------
# 2. Path to your input audio file
# --------------------------
audio_path = r"WhatsApp Audio 2025-11-16 at 19.34.41_eed179bd.mp3"

# --------------------------
# 3. Convert audio to WAV 16kHz mono (required for Vosk)
# --------------------------
sound = AudioSegment.from_file(audio_path)
sound = sound.set_frame_rate(16000).set_channels(1)
wav_path = "temp.wav"
sound.export(wav_path, format="wav")

# --------------------------
# 4. Transcribe using Vosk
# --------------------------
recognizer = KaldiRecognizer(model, 16000)

with open(wav_path, "rb") as f:
    while True:
        data = f.read(4000)
        if len(data) == 0:
            break
        recognizer.AcceptWaveform(data)

result = json.loads(recognizer.FinalResult())
hindi_text = result["text"]

print("🎤 Extracted Hindi Text:")
print(hindi_text)

with open("output.txt", "a", encoding="utf-8") as file:
     file.write(hindi_text)

from sarvamai import SarvamAI

# Load environment variables
load_dotenv()
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

# Initialize SarvamAI client (optional if you prefer SDK)
client = SarvamAI(api_subscription_key=SARVAM_API_KEY)

# Define supported languages (not used yet, but handy for prompts)
LANGUAGES = {
    "English": "en-IN",
    "Hindi": "hi-IN",
    "Gujarati": "gu-IN",
    "Bengali": "bn-IN",
    "Kannada": "kn-IN",
    "Punjabi": "pa-IN"
}

# Prepare request headers
headers = {
    "Authorization": f"Bearer {SARVAM_API_KEY}",
    "Content-Type": "application/json"
}

# Chat payload with correct message format
payload = {
    "model": "sarvam-m",
    "messages": [
        {"role": "user", "content": "hi i am sarthak"}
    ],
    # Optional parameters for more control:
    # "temperature": 0.7,
    # "reasoning_effort": "medium"
}

# Make the API request
response = requests.post(
    "https://api.sarvam.ai/v1/chat/completions",
    headers=headers,
    json=payload
)

# Handle response
# if response.status_code == 200:
#     assistant_reply = response.json()

# Extract only the assistant's text
    # assistant_text = assistant_reply["choices"][0]["message"]["content"]

    # Translate that text
translation = client.text.translate(
    input=hindi_text,
    source_language_code="hi-IN",
    target_language_code="en-IN",
    speaker_gender="Male"
)

final_reply = translation.translated_text
with open("output.txt" , "a" , encoding= "utf-8") as file:
    file.write(final_reply)

final = time.time()

print(f"Total Time = {final - start} seconds")


# --------------------------
# 5. (Optional) Send to LLM
# For now, just echo the text back
# --------------------------
llm_response = hindi_text  # Replace with actual LLM call

# --------------------------
# 6. Convert Hindi LLM response to Speech using gTTS
# --------------------------
# tts = gTTS(text=llm_response, lang='hi')
# output_audio = "response.mp3"
# tts.save(output_audio)

# print("\n🔊 Saved Hindi voice output to:", output_audio)

# # --------------------------
# # 7. Play the audio (Windows)
# # --------------------------
# os.system(f'start {output_audio}')   # For Windows

