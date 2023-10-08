from transformers import AutoTokenizer
import whisper 

class GetTextFromAudio: 
    def __init__(self) -> None:
        self.model = whisper.load_model("medium")
    def get_speech(self, audio) -> str:
        return self.model.transcribe(audio, fp16=False, language='en')['text']

class TokenizeText:
    def __init__(self, model_name='distilbert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    def tokenize(self, text):
        return self.tokenizer(text, return_tensors="pt")
