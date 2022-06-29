# ODE Structure against PGD ASR Attack
import os, sys
import time
import torch
import random
import numpy as np
import IPython.display as ipd
import matplotlib.pyplot as plt
from jiwer import wer,cer
from deepspeech_pytorch.loader.data_loader import load_audio
from art.estimators.speech_recognition import PyTorchDeepSpeech
from art.attacks.evasion.imperceptible_asr.imperceptible_asr_pytorch import ImperceptibleASRPyTorch
from art.attacks.evasion import CarliniWagnerASR
from art.defences.preprocessor import  Mp3Compression,GaussianAugmentation,LabelSmoothing,FeatureSqueezing,Resample
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

# Parameters initilaization
wer_max     = 0
at_wer_list = []
ao_wer_list = []
at_wer_num  = 0
ao_wer_num  = 0

# Set attack path 

data_wav_dire = '/data/RandomAudios2'
data_txt_dire = '/data/RandomTxts2'
adv_wav_dire  = '/data/Adv_Wav2'
adv_txt_dire  = '/data/Adv_Txt2'

# Sort audios files
wav_data_dire = os.listdir(data_wav_dire)
wav_data_dire.sort(key=lambda x:int(x.split('.')[0]))

# Sort txt files
txt_data_dire = os.listdir(data_txt_dire)
txt_data_dire.sort(key=lambda x:int(x.split('.')[0]))
wav_list_len = len(txt_data_dire) # wav file has the same length as txt file
#print('The length of nature wav/txt list is ',wav_list_len )

# Sort adversarial attack files
txt_adv_dire = os.listdir(adv_txt_dire)
txt_adv_dire.sort(key=lambda x:int((x.split('.')[0]).split('_')[1]))
wav_adv_dire = os.listdir(adv_wav_dire)
wav_adv_dire.sort(key=lambda x:int((x.split('.')[0]).split('_')[1]))

# ODE parameter initialization
T          = 1         # the endtime of processing. 
non_linear = torch.sin
coeffi     = -1        # -1 by default
layernum   = 0         # 0  by default
tol        = 1e-3      # 1e-3 by default

# Define ODE Bolck Part
class Fully_Connect(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Fully_Connect, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
    def forward(self, t, x):
        return self._layer(x)

class ODE_Function(nn.Module):
    def __init__(self, dim):
        super(ODE_Function, self).__init__()
        self.fc1 = Fully_Connect(dim, 256)
        self.non1 = non_linear
        self.fc2 = Fully_Connect(256, 256)
        self.non2 = non_linear
        self.fc3 = Fully_Connect(256, dim)
        self.non3 = non_linear
        self.ncount = 0

    def forward(self, t, x):
        self.ncount += 1
        out = coeffi*self.fc1(t, x)
        out = self.non1(out)
        out = coeffi*self.fc2(t, out)
        out = self.non2(out)
        out = coeffi*self.fc3(t, out)
        out = self.non3(out)
        return out

class ODEBlock(nn.Module):
    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, T]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=tol, atol=tol)
        return out

    @property
    def ncount(self):
        return self.odefunc.ncount

    @ncount.setter
    def ncount(self, value):
        self.odefunc.ncount = value
        
        
# Create a DeepSpeech estimator
speech_recognizer = PyTorchDeepSpeech(pretrained_model="tedlium")
labels_map = dict([(speech_recognizer.model.labels[i], i) for i in range(len(speech_recognizer.model.labels))])
def parse_transcript(path):
    with open(path, 'r', encoding='utf8') as f:
        transcript = f.read().replace('\n', '')
    result = list(filter(None, [labels_map.get(x) for x in list(transcript)]))
    return transcript, result

# SNR caculation
def mySNR(clean,output):
    length = min(len(clean),len(output))
    noise = output[:length] - clean[:length]
    snr = 10*np.log10((np.sum(clean**2))/(np.sum(noise**2)))
    print(snr)

# WER caculation
def myWER(Groundtrue_txt, Predicted_txt):
    g_str = ''.join(str(ig) for ig in Groundtrue_txt)
    p_str = ''.join(str(ip) for ip in Predicted_txt)
    error = wer(g_str, p_str)
    return error

def Datainfo(wav_index):
    print('The type of loaded wav audio is ',type([wav_index]))
    print('The size of loaded wav audio is ',len(wav_index))
    print('The loaded wav audio is ',wav_index)
    #print('The shape of loaded wav audio is ',shape(wav_index))

def NpDatainfo(np_wav_index):
    print('The type of loaded numpy wav audio is ',type(np_wav_index))
    print('The size of loaded numpy wav audio is ',np_wav_index.size)
    print('The shape of loaded numpy wav audio is ',np_wav_index.shape)
    print('The dimension of loaded numpy wav audio is ',np_wav_index.ndim)
    print('The loaded numpy wav audio is ',np_wav_index)


# main function
num_loop = 2 # wav_list_len can be set randomly.

for n in range(0, num_loop):
    #speech_recognizer_defense = PyTorchDeepSpeech(pretrained_model="tedlium",preprocessing_defences=None)
    # Load clean txt files 
    label_index, encoded_label_index = parse_transcript(os.path.join(data_txt_dire,txt_data_dire[n]))
    print("The groundtruth is : ", label_index)

    # Load adv audio for PGD Attack
    wav_index = load_audio(os.path.join(adv_wav_dire,wav_adv_dire[n]))
    adv_load =  [wav_index]
    
    # Handle Nan value
    adv_load[0][np.isnan(adv_load[0])]=1e-5
    # Transcript the adversial samples
    attack_adv_transcription = speech_recognizer.predict(np.array(adv_load), transcription_output=True)
    print("Without any defenses, the adversarial samples transcripted by deepspeech: ", attack_adv_transcription[0])

    # Load adversarial txt files.
    adv_label_index, adv_encoded_label_index = parse_transcript(os.path.join(adv_txt_dire,txt_adv_dire[n]))
    print('The adversarial target of {ni}-th audio should to be {label}'.format(ni=n,label=adv_label_index))
    
    #ODE defense part
    odefunc = ODE_Function((adv_load[0]).size)
    feature_layers = ODEBlock(odefunc)
    output = feature_layers(torch.from_numpy(adv_load[0]))
    npout  = output.detach().numpy()
    newout = [1e-3*npout[0]+1e-6*npout[1]]
    #NpDatainfo(np.array(newout))
    
    adv_transcription = speech_recognizer.predict(np.array(newout), transcription_output=True)
    print("After adding the ODE defense block, the transcription turns to be: ", adv_transcription[0])

    #adv txt load
    #adv_label_index, adv_encoded_label_index = parse_transcript(os.path.join(adv_txt_dire,txt_adv_dire[n]))
    #print('The adv target of {ni}th audio should to be {label}'.format(ni=n,label=adv_label_index))
    #print('The WER between adv txt and target txt is as follow ')
    #myWER(adv_label_index, adv_transcription[0])
    
    #SNR only subject to single sentence
    #print('The {ni}-th SNR as follow: '.format(ni=n))
    #mySNR(np.array(wav_index),np.array(adv_load))

    # Caculate WER average and max items
    at_wer_num = myWER(adv_label_index, adv_transcription[0])
    ao_wer_num = myWER(label_index, adv_transcription[0])
    at_wer_list.append(at_wer_num)
    ao_wer_list.append(ao_wer_num)

# List to ndarray
at_wer_list=np.array(at_wer_list)
print('at_wer_list is',at_wer_list)
ao_wer_list=np.array(ao_wer_list)
print('ao_wer_list is',ao_wer_list)

# Max and average value from at_wer_list between adv and target
at_wer_max=np.max(at_wer_list)
print('After adding Defenses,the max wer value between adv and target is ',at_wer_max)
at_wer_max_index=np.where(at_wer_list==at_wer_max)
print('After adding Defenses,the max wer location between adv and target is ',at_wer_max_index)
print('After adding Defenses,the average wer value between adv and target is ',np.mean(at_wer_list))
# Max and average value from ao_wer_list between adv and original
ao_wer_max=np.max(ao_wer_list)
print('After adding Defenses,the max wer value between adv and ori is ',ao_wer_max)
ao_wer_max_index=np.where(ao_wer_list==ao_wer_max)
print('After adding Defenses,the max wer location between adv and ori is ',ao_wer_max_index)
print('After adding Defenses,the average wer value between adv and ori is ',np.mean(ao_wer_list))
