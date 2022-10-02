#@title Choose English ASR model { run: "auto" }

lang = 'en'
fs = 16000 #@param {type:"integer"}
tag = 'Shinji Watanabe/spgispeech_asr_train_asr_conformer6_n_fft512_hop_length256_raw_en_unnorm_bpe5000_valid.acc.ave' #@param ["Shinji Watanabe/spgispeech_asr_train_asr_conformer6_n_fft512_hop_length256_raw_en_unnorm_bpe5000_valid.acc.ave", "kamo-naoyuki/librispeech_asr_train_asr_conformer6_n_fft512_hop_length256_raw_en_bpe5000_scheduler_confwarmup_steps40000_optim_conflr0.0025_sp_valid.acc.ave"] {type:"string"}

import time
import torch
import string
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.asr_inference import Speech2Text


d = ModelDownloader()
# It may takes a while to download and build models
speech2text = Speech2Text(
    **d.download_and_unpack(tag),
    device="cuda",
    minlenratio=0.0,
    maxlenratio=0.0,
    ctc_weight=0.3,
    beam_size=10,
    batch_size=0,
    nbest=1
)

def text_normalizer(text):
    text = text.upper()
    return text.translate(str.maketrans('', '', string.punctuation))

# !git clone https://github.com/ftshijt/ESPNet_asr_egs.git

# import pandas as pd
# import soundfile
# import librosa.display
# from IPython.display import display, Audio
# import matplotlib.pyplot as plt


# egs = pd.read_csv("ESPNet_asr_egs/egs.csv")
# for index, row in egs.iterrows():
#   if row["lang"] == lang or lang == "multilingual":
#     speech, rate = soundfile.read("ESPNet_asr_egs/" + row["path"])
#     assert fs == int(row["sr"])
#     nbests = speech2text(speech)

#     text, *_ = nbests[0]
#     print(f"Input Speech: ESPNet_asr_egs/{row['path']}")
#     # let us listen to samples
#     display(Audio(speech, rate=rate))
#     librosa.display.waveplot(speech, sr=rate)
#     plt.show()
#     print(f"Reference text: {text_normalizer(row['text'])}")
#     print(f"ASR hypothesis: {text_normalizer(text)}")
#     print("*" * 50)

# from https://gist.github.com/korakot/c21c3476c024ad6d56d5f48b0bca92be

from IPython.display import Javascript
# from google.colab import output
from base64 import b64decode

RECORD = """
const sleep = time => new Promise(resolve => setTimeout(resolve, time))
const b2text = blob => new Promise(resolve => {
  const reader = new FileReader()
  reader.onloadend = e => resolve(e.srcElement.result)
  reader.readAsDataURL(blob)
})
var record = time => new Promise(async resolve => {
  stream = await navigator.mediaDevices.getUserMedia({ audio: true })
  recorder = new MediaRecorder(stream)
  chunks = []
  recorder.ondataavailable = e => chunks.push(e.data)
  recorder.start()
  await sleep(time)
  recorder.onstop = async ()=>{
    blob = new Blob(chunks)
    text = await b2text(blob)
    resolve(text)
  }
  recorder.stop()
})
"""

def record(sec, filename='audio.wav'):
  display(Javascript(RECORD))
#   s = output.eval_js('record(%d)' % (sec * 1000))
  b = b64decode(s.split(',')[1])
  with open(filename, 'wb+') as f:
    f.write(b)

audio = './my_data1.wav'
# second = 5
# print(f"Speak to your microphone {second} sec...")
# record(second, audio)
# print("Done!")


import librosa
import librosa.display
speech, rate = librosa.load(audio, sr=16000)
# librosa.display.waveplot(speech, sr=rate)

import matplotlib.pyplot as plt
plt.show()

# import pysndfile
# pysndfile.sndio.write('audio_ds.wav', speech, rate=rate, format='wav', enc='pcm16')

from IPython.display import display, Audio
display(Audio(speech, rate=rate))

nbests = speech2text(speech)
text, *_ = nbests[0]

print(f"ASR hypothesis: {text_normalizer(text)}")