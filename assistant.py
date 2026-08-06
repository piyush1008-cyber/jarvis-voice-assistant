import os
import requests
import pyttsx3
import threading
import sounddevice as sd
from scipy.io import wavfile
import speech_recognition as sr

# Local Ollama endpoint
OLLAMA_URL = "http://localhost:11434/api/generate"

def run_tts(text):
    """
    Initializes pyttsx3 and speaks the given text.
    Runs inside a separate thread to prevent blocking the UI.
    """
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 175) # Set speech speed
        
        # Look for a male voice (Jarvis style)
        voices = engine.getProperty('voices')
        if voices:
            # Typically voices[0] is male (David/default), voices[1] is female (Zira)
            engine.setProperty('voice', voices[0].id)
            
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"Text-to-Speech Error: {e}")

def speak(text):
    """
    Wrapper to run TTS asynchronously in a background thread.
    """
    t = threading.Thread(target=run_tts, args=(text,))
    t.start()

def record_audio(duration=6, sample_rate=16000, filename="temp_query.wav"):
    """
    Records audio from the default microphone using sounddevice.
    Avoids PyAudio dependency issues on Windows.
    """
    try:
        print(f"Recording voice for {duration} seconds...")
        # Record mono channel 16-bit audio
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait() # Wait until recording is completed
        print("Recording complete.")
        
        # Save recording to WAV file
        wavfile.write(filename, sample_rate, recording)
        return True
    except Exception as e:
        print(f"Audio Recording Error: {e}")
        return False

def transcribe_audio(filename="temp_query.wav"):
    """
    Transcribes the recorded WAV file using SpeechRecognition (Google Web Speech API).
    """
    if not os.path.exists(filename):
        return ""
        
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(filename) as source:
            # Adjust for ambient noise and record
            audio_data = recognizer.record(source)
            print("Transcribing audio...")
            text = recognizer.recognize_google(audio_data)
            return text
    except sr.UnknownValueError:
        print("Speech recognition could not understand audio.")
        return ""
    except sr.RequestError as e:
        print(f"Speech recognition service error: {e}")
        return ""
    except Exception as e:
        print(f"Transcription Error: {e}")
        return ""

def query_ollama(prompt_text, model_name="llama3.2"):
    """
    Sends a query prompt to local Ollama server running Llama-3.2.
    """
    payload = {
        "model": model_name,
        "prompt": prompt_text,
        "stream": False
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=20)
        if response.status_code == 200:
            return response.json().get('response', '').strip()
        else:
            return f"Error connecting to Ollama: Code {response.status_code}"
    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to Ollama. Please make sure the Ollama application is running (`ollama run llama3.2`)."
    except Exception as e:
        return f"Error communicating with Ollama: {str(e)}"

def format_jarvis_prompt(question, emotion, context=None):
    """
    Constructs the custom prompt for Llama-3.2 incorporating context and emotion.
    """
    prompt = f"""You are Jarvis, an advanced, highly intelligent AI companion. The user is asking you a question.
The user is currently feeling: {emotion}.

User's Question: "{question}"
"""
    if context:
        prompt += f"""
Here is the relevant context retrieved from their uploaded PDF document to answer the question:
---
{context}
---
Instructions:
1. Rely primarily on the provided PDF context to answer the user.
2. Keep your answer brief, direct, and conversational (2 to 3 sentences maximum), suitable for voice playback.
3. Dynamically adjust your tone of voice to match, comfort, or respond to the user's emotion ({emotion}).
"""
    else:
        prompt += f"""
Instructions:
1. Provide a brief, clever, and helpful response (2 to 3 sentences maximum), suitable for voice playback.
2. Dynamically adjust your tone of voice to match, comfort, or respond to the user's emotion ({emotion}).
"""
    return prompt
