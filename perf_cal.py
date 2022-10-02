#@title Choose English ASR model { run: "auto" }

lang = 'en'
fs = 16000 #@param {type:"integer"}
tag = 'Shinji Watanabe/spgispeech_asr_train_asr_conformer6_n_fft512_hop_length256_raw_en_unnorm_bpe5000_valid.acc.ave' #@param ["Shinji Watanabe/spgispeech_asr_train_asr_conformer6_n_fft512_hop_length256_raw_en_unnorm_bpe5000_valid.acc.ave", "kamo-naoyuki/librispeech_asr_train_asr_conformer6_n_fft512_hop_length256_raw_en_bpe5000_scheduler_confwarmup_steps40000_optim_conflr0.0025_sp_valid.acc.ave"] {type:"string"}

from email.mime import audio
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

from IPython.display import Javascript
from base64 import b64decode

import librosa
import librosa.display
import matplotlib.pyplot as plt
from IPython.display import display, Audio
import sys


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
total = 0
err = 0
f = open("./84-121123.trans.txt", 'r')
data = f.readlines()
for line in data:
    total += 1
    # print(line)
    line_list = line.split(' ') 
    forder_address = line_list[0].split('-')

    # print(line[len(line_list[0])+1:])
    f1 = forder_address[0]
    f2 = forder_address[1]
    fn = line_list[0]
    
    audio = './LibriSpeech/dev-clean/'+f1+'/'+f2+'/'+fn+'.flac'
    speech, rate = librosa.load(audio, sr=16000)
    display(Audio(speech, rate=rate))
    nbests = speech2text(speech)
    text, *_ = nbests[0]
    
    print(line[len(line_list[0])+1:].rstrip())
    print(text_normalizer(text))

    if line[len(line_list[0])+1:].rstrip() != text_normalizer(text):
        err += 1
        print(err)

    print()

print('err / total_dataset : '+str(err)+' / '+str(total))

    # print(len(line_list[0]))
    # print(forder_address)

# print(f.readlines())

# for i in range(4):

# audio = './my_data1.wav'


# print(f"ASR hypothesis: {text_normalizer(text)}")