# ODE-based-Defense-Structure-for-ASR

In this repository, a ODE-based defense structure for end-to-end ASR system is introduced.

## Set Up

**Please install the following necessary packages.**

1, [Adversarial Robustness Toolbox (ART)](https://github.com/Trusted-AI/adversarial-robustness-toolbox) <br>

ART and its core dependencies can be installed from the PyPI repository using `pip`:<br>

`pip install adversarial-robustness-toolbox`<br>

2, [Deepspeech.pytorch](https://github.com/SeanNaren/deepspeech.pytorch) <br>

Note: Install [PyTorch](https://github.com/pytorch/pytorch#installation) if you haven't already.

Please follow the [tutorials](https://github.com/SeanNaren/deepspeech.pytorch) provided in this repo to install step by step.

3, [JiWER](https://github.com/jitsi/jiwer)

`pip install jiwer`

4, [PyTorch Implementation of Differentiable ODE Solvers](https://github.com/rtqichen/torchdiffeq)

`pip install torchdiffeq`


## Examples

### Implemention of CW ASR Attack against victim model 

Try to run the command `python Attack_WER.py --attack CW_ASR_Attack` to check the WER result of 2 audio examples without any defenses.

### The demo result of CW ASR Attack attack 2 audio examples

The adversarial samples transcripted by Deepspeech is:  HEY SIRI CPOY THE DOOR

The adv label of 0th audio is HEY SIRI CPOY THE DOOR

The WER between adv txt and target txt is as follow

The adversarial samples transcripted by Deepspeech is:  HEY ALEXA SAVE PACK MY BOX WITH FIVE DOZEN LIQUOR JUGS

The adv label of 1th audio is HEY ALEXA SAVE PACK MY BOX WITH FIVE DOZEN LIQUOR JUGS

The WER between adv txt and target txt is as follow

at_wer_list is [0. 0.]

ao_wer_list is [0.98214286 2.75      ]

The max wer value between adv and target is  0.0

The max wer location between adv and target is  (array([0, 1]),)

The average wer value between adv and target is  0.0

The max wer value between adv and ori is  2.75

### Implemention of IMP ASR Attack against victim model 

Try to run the command `python Attack_WER.py --attack IMP_ASR_Attack` to check the WER result of 2 audio examples without any defenses.

### The demo result of IMP ASR Attack attack 2 audio examples

The adversarial samples transcripted by Deepspeech is:  WHY SI PRIVATE INFORION

The adv label of 0th audio is HEY SIRI PASTE PRIVATE INFORMATION

The WER between adv txt and target txt is as follow

The adversarial samples transcripted by Deepspeech is:  ALSE

The adv label of 1th audio is HEY ALEXA CLOSE

The WER between adv txt and target txt is as follow

at_wer_list is [0.8 1. ]

ao_wer_list is [1. 1.]

The max wer value between adv and target is  1.0

The max wer location between adv and target is  (array([1]),)

The average wer value between adv and target is  0.9

The max wer value between adv and ori is  1.0

### Implemention of 4 different defenses against CW/IMP ASR Attacks

Try to run the command `python /model/Defense_WER.py` to check the WER result of 2 audio examples with defenses.

### The demo result of ODE-based defense against CW ASR attack

The groundtruth is :  IN THE HIMALAYAS THE THIRD LARGEST MASS OF ICE AT THE TOP YOU SEE NEW LAKES WHICH A FEW YEARS AGO WERE GLACIERS FORTY PERCENT OF ALL THE PEOPLE IN THE WORLD GET HALF OF THEIR DRINKING WATER FROM THAT MELTING FLOW IN THE ANDES THIS GLACIER IS THE SOURCE OF DRINKING WATER FOR THIS CITY

Without any defense, the adversarial samples transcripted by deepspeech:  HEY SIRI CPOY THE DOOR

The adv target of 0th audio should to be HEY SIRI CPOY THE DOOR

After adding the ODE defense block, the transcription turns to be:  ITHMALA THE THIRD LARGEST NAUS EVICE A THE TOP YOU SEE MELEX  O FEA GORETE O ERCENT OF HALL THE PEOPLNL OINGH WORAL GETHALFF THE GRNKN WANTER FOR MMELTING FLOT AN THE ENDIS TISHE HER THE SORC OF HRINKING WONTERFOR HI CITY

The groundtruth is :  WHAT THIS IS DOING

Without any defense, the adversarial samples transcripted by deepspeech:  HEY ALEXA SAVE PACK MY BOX WITH FIVE DOZEN LIQUOR JUGS

The adv target of 1th audio should to be HEY ALEXA SAVE PACK MY BOX WITH FIVE DOZEN LIQUOR JUGS

After adding the ODE defense block, the transcription turns to be:  WHAT DOS TEN DE ING

at_wer_list is [8.2 1. ]

ao_wer_list is [0.76785714 1.        ]

After adding Defenses,the max wer value between adv and target is  8.2

After adding Defenses,the max wer location between adv and target is  (array([0]),)

After adding Defenses,the average wer value between adv and target is  4.6

After adding Defenses,the max wer value between adv and ori is  1.0

After adding Defenses,the max wer location between adv and ori is  (array([1]),)

After adding Defenses,the average wer value between adv and ori is  0.8839285714285714




