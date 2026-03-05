import speech_recognition as sr
import pyttsx3
from app.rag.query_engine import create_qa_chain

qa_chain = create_qa_chain()

recognizer = sr.Recognizer()
engine = pyttsx3.init()

def listen():
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)

        try:
            text = recognizer.recognize_google(audio)
            print("You said:", text)
            return text
        except Exception:
            print("Could not understand")
            return None

def speak(text):
    print("Assistant:", text)
    engine.say(text)
    engine.runAndWait()

def run_voice_assistant():
    while True:
        query = listen()

        if query is None:
            continue

        if "exit" in query.lower():
            speak("Goodbye")
            break

        response = qa_chain({"query": query})
        answer = response["result"]

        speak(answer)
