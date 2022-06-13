### result after being attacked by three different ASR attacks
import os, sys
#import time
import torch
import random
import numpy as np
import IPython.display as ipd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from jiwer import wer,cer
from deepspeech_pytorch.loader.data_loader import load_audio
from art.estimators.speech_recognition import PyTorchDeepSpeech
from art.attacks.evasion.imperceptible_asr.imperceptible_asr_pytorch import ImperceptibleASRPyTorch
from art.attacks.evasion import CarliniWagnerASR
from art.defences.preprocessor import GaussianAugmentation,LabelSmoothing,Resample

# Parameters initilaization
wer_max = 0
at_wer_list=[]
ao_wer_list=[]
at_wer_num=0
ao_wer_num=0
attack_flag = ''
defense_flag = 0

parser = ArgumentParser()
parser.add_argument('--attack', 
        type = str,
        default = 'IMP_ASR_Attck',
        help = 'Method of adversarial attack')
parser.add_argument('--defense',
        type = str,
        help = 'Method of defense')


#Defenses Part

DOWNSAMPLED_SAMPLING_RATE=16000

#Prepossing defense1 - Gaussian noise
gaussian = GaussianAugmentation(sigma = 1.0, augmentation = False, apply_fit = True, apply_predict = False)

#Prepossing defense2 - LabelSmoothing
smooth = LabelSmoothing(max_value = 0.9, apply_fit = True, apply_predict = False)

#Prepossing defense3 - Resample
RS = Resample(sr_original=DOWNSAMPLED_SAMPLING_RATE,sr_new=DOWNSAMPLED_SAMPLING_RATE,channels_first = True,apply_fit = True,apply_predict = False)

args = parser.parse_args()

if args.attack == 'IMP_ASR_Attack':
    attack_flag = ''

if args.attack == 'CW_ASR_Attack':
    attack_flag = '1'

if args.attack == 'PGD_ASR_Attack':
    attack_flag = '2'

if args.defense == 'Gaussian':
    defense_flag = 1
    defense = gaussian

if args.defense == 'Smooth':
    defense_flag = 2
    defense = smooth

if args.defense == 'Resample':
    defense_flag = 3
    defense = RS




# Set attack methods path
data_wav_dire = '/data/home/wentao/adversarial-robustness-toolbox/taotest/RandomAudios' + attack_flag
data_txt_dire = '/data/home/wentao/adversarial-robustness-toolbox/taotest/RandomTxts' + attack_flag
adv_wav_dire = '/data/home/wentao/adversarial-robustness-toolbox/taotest/Adv_Wav' + attack_flag
adv_txt_dire = '/data/home/wentao/adversarial-robustness-toolbox/taotest/Adv_Txt' + attack_flag
#data_wav_dire = '/data/RandomAudios'
#data_txt_dire = '/data/RandomTxts'
#adv_wav_dire = '/data/Adv_Wav'
#adv_txt_dire = '/data/Adv_Txt'

# Sort audio files 
wav_data_dire = os.listdir(data_wav_dire)
wav_data_dire.sort(key=lambda x:int(x.split('.')[0]))

# Sort txt files
txt_data_dire = os.listdir(data_txt_dire)
txt_data_dire.sort(key=lambda x:int(x.split('.')[0]))

# Get the length of audio files, audio file has the same length as txt file
wav_list_len = len(wav_data_dire) 

# Sort adversarial attack files
txt_adv_dire = os.listdir(adv_txt_dire)
txt_adv_dire.sort(key=lambda x:int((x.split('.')[0]).split('_')[1]))
wav_adv_dire = os.listdir(adv_wav_dire)
wav_adv_dire.sort(key=lambda x:int((x.split('.')[0]).split('_')[1]))

# Create a DeepSpeech estimator
if defense_flag == 0:
    speech_recognizer = PyTorchDeepSpeech(pretrained_model="tedlium")
else:
    speech_recognizer = PyTorchDeepSpeech(pretrained_model="tedlium",preprocessing_defences=defense)

# Parse the transcript recoginized by DeepSpeech.
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
    length = min(len(clean),len(output))
    noise = output[:length] - clean[:length]
    snr = 10*np.log10((np.sum(clean**2))/(np.sum(noise**2)))
    print(snr)

# WER
def myWER(Groundtrue_txt, Predicted_txt):
    g_str = ''.join(str(ig) for ig in Groundtrue_txt)
    p_str = ''.join(str(ip) for ip in Predicted_txt)
    error = wer(g_str, p_str)
    return error


# Main function
num_loop=2 #wav_list_len
for n in range(0,num_loop):
    
    # clean txt load
    label_index, encoded_label_index = parse_transcript(os.path.join(data_txt_dire,txt_data_dire[n]))
    
    # adversarial audios load for IMP and CW Attack
    new_attacked_name = 'adv'+ attack_flag + '_' + str(n)
    adv_load = np.load(adv_wav_dire + '/' + new_attacked_name +'.npy')
    
    #(r'/data/Adv_Wav/'+new_attacked_name+'.npy')
    adv_transcription = speech_recognizer.predict(np.array(adv_load), transcription_output=True)
    print("The adversarial samples transcripted by Deepspeech is: ", adv_transcription[0])
    
    # PGD Attack
    #wav_index = load_audio(os.path.join(adv_wav_dire,wav_adv_dire[n]))
    #adv_transcription = speech_recognizer.predict(np.array([wav_index]), transcription_output=True)
    #print("Adversarial transcriptions transcripted by deepspeech: ", adv_transcription[0])

    # adv txt load
    adv_label_index, adv_encoded_label_index = parse_transcript(os.path.join(adv_txt_dire,txt_adv_dire[n]))
    print('The adv label of {ni}th audio is {label}'.format(ni=n,label=adv_label_index))
    print('The WER between adv txt and target txt is as follow ')
    myWER(adv_label_index, adv_transcription[0])
    

    #SNR only subject to single sentence
    #print('The {ni}-th SNR as follow: '.format(ni=n))
    #mySNR(np.array(wav_index),np.array(adv_load))
    
    #WER Average and Max value 
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
print('The max wer value between adv and target is ',at_wer_max)
at_wer_max_index=np.where(at_wer_list==at_wer_max)
print('The max wer location between adv and target is ',at_wer_max_index)
print('The average wer value between adv and target is ',np.mean(at_wer_list))
# Max and average value from ao_wer_list between adv and original
ao_wer_max=np.max(ao_wer_list)
print('The max wer value between adv and ori is ',ao_wer_max)
ao_wer_max_index=np.where(ao_wer_list==ao_wer_max)
