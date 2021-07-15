import speech_recognition as sr
import pyaudio
import sys
import io
import time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')

class my_record(object):
    def __init__(self):
        self.sample_rate = 16000
        self.device_index = 1
    def begin_record(self):
        r = sr.Recognizer()
        with sr.Microphone(sample_rate = self.sample_rate, device_index = 0) as source:
            r.energy_threshold = 300
            r.dynamic_energy_threshold = True
            r.pause_threshold = 1
            print("speak")
            audio = r.listen(source)
        with open("/home/fyeward/Desktop/api/test.pcm", 'wb') as f:
            f.write(audio.get_wav_data())
        print("finish")


a = my_record()
a.begin_record()