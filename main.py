import numpy as np
import speech_recognition as sr
import transformers
from gtts import gTTS
import os
import datetime
import time
from transformers import GPT2Tokenizer, TFGPT2Model

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = TFGPT2Model.from_pretrained('gpt2')
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='tf')
output = model(encoded_input)
#Building the AI
class ChatBot():
    def __init__(self,name):
        print("----- starting up", name, "-----")
        self.name = name
    def speech_to_text(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as mic:
            print("listening...")
            audio = recognizer.listen(mic)
        try:
            self.text = recognizer.recognize_google(audio)
            print("me --> ", self.text)
        except:
            print("me --> ERROR")
    @staticmethod
    def text_to_speech(text):
        print("Dev --> ", text)
        speaker = gTTS(text=text.lower(), lang="en", slow=False)
        speaker.save("res.mp3")
        statbuf = os.stat("res.mp3")
        mbytes = statbuf.st_size / 1024
        duration = mbytes / 200
        os.system("start res.mp3")
        time.sleep(int(50*duration))
        os.remove("res.mp3")
    def wake_up(self, text):
        return True if self.name in text else False
    @staticmethod
    def action_time():
        return datetime.datetime.now().time().strftime('%H:%M')

# Execute the AI
if __name__ == "__main__":
    ai = ChatBot(name="Dev")
    nlp = transformers.pipeline("conversational", model=TFGPT2Model.from_pretrained('gpt2'))
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    ex=True
    while ex:
        ai.speech_to_text()
        # wake up
        if ai.wake_up(ai.text) is True:
            res = "Hello I am Dev the AI, what can I do for you?"
        # tells time
        elif "time" in ai.text:
            res = ai.action_time()
        #respond politely
        elif any(i in ai.text for i in ["thank", "thanks"]):
            res = np.random.choice(
                ["you're welcome!", "anytime!", "no problem!", "cool!", "of course!", "I'm here if you need me!", "later nerd!"])
        elif any(i in ai.text for i in ["exit", "close"]):
            res = np.random.choice(["Tata", "Have a good day", "Bye", "Goodbye", "Later nerd."])
            ex=False
        else:
            if ai.text=="ERROR":
                res="Sorry, come again?"
            else:
                chat = nlp(transformers.Conversation(ai.text), pad_token_id=50256)
                res = str(chat)
                res = res[res.find("bot >> ") + 6:].strip()
        ai.text_to_speech(res)