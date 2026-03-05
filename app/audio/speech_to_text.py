import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import tempfile

# load model once
model = whisper.load_model("base")


def record_audio(duration=5):
    import sounddevice as sd
    from scipy.io.wavfile import write
    import tempfile

    print("🎤 Recording... Speak now")

    DEVICE_INDEX = 0  # MacBook Air Microphone

    samplerate = int(sd.query_devices(DEVICE_INDEX, 'input')['default_samplerate'])

    recording = sd.rec(
        int(duration * samplerate),
        samplerate=samplerate,
        channels=1,
        dtype='int16',
        device=DEVICE_INDEX
    )
    sd.wait()

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    write(temp_file.name, samplerate, recording)

    print("✅ Recording complete")

    return temp_file.name


def transcribe_audio(audio_path):
    print("🧠 Transcribing...")

    result = model.transcribe(audio_path)

    text = result["text"]
    print(f"📝 You said: {text}")

    return text


if __name__ == "__main__":
    path = record_audio()
    text = transcribe_audio(path)