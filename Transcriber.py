import re
import whisper
import SetupChecker


class Transcriber:
    def __init__(self, language="de"):
        self.language = language
        self.model = whisper.load_model("tiny", device="cuda")
    def transcribe(self, filename):
        print(f"TRANSCRIBING {filename}")
        result = self.model.transcribe(audio = filename, language = self.language)
        result = result["text"]

        sentences = re.split(r'(?<=[.!?])\s+', result)
        for sentence in sentences:
            print(sentence.strip())
        print("DONE")
        print("--------------------------------------------------------")

        return result


if __name__ == "__main__":
    # SetupChecker.check()
    t = Transcriber()
    transcript = t.transcribe(filename = "test_audio.mp3")