import os
import time
import json
import pyttsx3
import speech_recognition as sr
from vosk import Model, KaldiRecognizer

# 1. Initialize Text-to-Speech
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)

# Fetch all available macOS native voices
voices = tts_engine.getProperty('voices')

# Find a German voice
for voice in voices:  # type: ignore
    # macOS voice IDs for German usually contain 'de_DE' or 'de_'
    if 'de_' in voice.id.lower() or 'de_de' in str(voice.languages).lower():
        tts_engine.setProperty('voice', voice.id)
        print(f"Success: Set TTS voice to {voice.name} (German)")
        break

def speak(text):
    print(f"\n[System]: {text}")
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.say(text)
    engine.runAndWait()
    engine.stop()

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "models")

print(f"Loading offline language model from: {model_path}")
try:
    offline_model = Model(model_path)
except Exception as e:
    print(f"\n[ERROR] Failed to load model. Please ensure the 'model' folder exists at {model_path}")
    exit(1)

def listen_and_respond():
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 400
    recognizer.dynamic_energy_threshold = False
    print("\nListening for speech...")
    
    with sr.Microphone() as source:
        try:
            # Capture the audio stream
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            print("Processing audio locally...")
            
            vosk_recognizer = KaldiRecognizer(offline_model, 16000)
            raw_audio_data = audio.get_raw_data(convert_rate=16000, convert_width=2)
            
            # Process the audio
            vosk_recognizer.AcceptWaveform(raw_audio_data)
            result_json = vosk_recognizer.FinalResult()
            
            # Extract text
            data = json.loads(result_json)
            text = data.get("text", "")
            
            if text:
                print(f"[You Said]: {text}")
                
                if "hallo" in text:
                    speak("Hallo! Wie geht es dir?")
                elif "test" in text:
                    speak("Das hier ist ein Test")
                elif "stop" in text or "exit" in text:
                    speak("Ich bin Müde, bis zum nächsten Mal!")
                    time.sleep(1)
                    return False
                else:
                    speak(f"Ich habe {text} verstanden, aber ich habe keine spezifische Antwort dafür.")
            
        except sr.WaitTimeoutError:
            pass
        except Exception as e:
            print(f"An error occurred: {e}")
            
    return True

if __name__ == "__main__":
    speak("System initialisiert.")
    
    running = True
    while running:
        running = listen_and_respond()