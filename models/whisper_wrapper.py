from thirdparty.whisper import whisper

class WhisperTranscriber:
    def __init__(self, model_name="base"):
        self.model = whisper.load_model(model_name)

    def transcribe(self, audio_path: str) -> str:
        result = whisper.transcribe(self.model, audio_path)
        return result.get("text", "").strip()

if __name__ == "__main__":
    transcriber = WhisperTranscriber()
    text = transcriber.transcribe("./../data/test/test_audio.wav")
    print("Transcribed:", text)
