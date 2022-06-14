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

## Warm-up 

AO AVERAGE is average WER between defended examples and original audios
to test how severe the damage is after being attacked. The smaller the value,
the stronger the defense. The defended example represents the defended attack
instance, i.e., the original clean speech is first attacked to produce the corresponding
adversarial sample, and then the corresponding defense method is applied to
this sample.

AO MAX is max WER between defended examples and original audios. The
smaller the value, the stronger the defense.

AT MAX is max WER between defended examples and attack targets. Therefore,
the larger this value is, the further the two are from each other. That is, the less
similar they are.

AT AVERAGE is average WER between defended examples and attack targets
to test how far is the distance between the actual attack example crafted by the
attacker and its set attack target. AT AVERAGE has a similar meaning to AT
MAX, except that it is expressed as the average distance, i.e., it represents the
overall effect of all the data under this attack. The larger the value, the stronger
the defense.

## Adversarial Examples Generating

To save running time, only two randomly generated speechs from the open source corpus Tedlium were selected during this test.

### Implemention of [IMP ASR Attack](https://arxiv.org/abs/1903.10346) against ASR System DeepSpeech

First you can add a custom voice attack method to the corpus, i.e., please try to run the command `python model/Attack_WER.py --attack IMP_ASR_Attack` to check the WER of 2 speechs without any defenses.

#### If you run the command above correctly, you will get the following results:

The adversarial samples transcripted by Deepspeech is:  HEY SIRI CPOY THE DOOR

The adv label of 0th audio is HEY SIRI CPOY THE DOOR

The adversarial samples transcripted by Deepspeech is:  HEY ALEXA SAVE PACK MY BOX WITH FIVE DOZEN LIQUOR JUGS

The adv label of 1th audio is HEY ALEXA SAVE PACK MY BOX WITH FIVE DOZEN LIQUOR JUGS

The WER between adv txt and target txt is as follow

at_wer_list is [0. 0.]

ao_wer_list is [0.98214286 2.75      ]

The max wer value between adv and target is  0.0

The max wer location between adv and target is  (array([0, 1]),)

The average wer value between adv and target is  0.0

The max wer value between adv and original is  2.75

### Implemention of [CW ASR Attack](https://arxiv.org/abs/1801.01944) against ASR System DeepSpeech

Operate like above, try to run the command ` python model/Attack_WER.py --attack CW_ASR_Attack` to check the WER result of 2 audio examples without any defenses.


## Voice Defenses against Adversarial Examples

On the other hand, you can add commonly used defense methods against the adeversarial examples 

### Implemention Gaussian defense against IMP/CW ASR Attacks

Try to run the command `python model/Attack_WER.py --attack IMP_ASR_Attack --defense Gaussian` to check the WER result of 2 audio examples with defenses.

#### If you run the command above correctly, you will get the following results:

The adversarial samples transcripted by Deepspeech is:  HEY SIRI CPOY THE DOOR

The adv label of 0th audio is HEY SIRI CPOY THE DOOR

The adversarial samples transcripted by Deepspeech is:  HEY ALEXA SAVE PACK MY BOX WITH FIVE DOZEN LIQUOR JUGS

The adv label of 1th audio is HEY ALEXA SAVE PACK MY BOX WITH FIVE DOZEN LIQUOR JUGS

The WER between adv txt and target txt is as follow

at_wer_list is [0. 0.]

ao_wer_list is [0.98214286 2.75      ]

The max wer value between adv and target is  0.0

The max wer location between adv and target is  (array([0, 1]),)

The average wer value between adv and target is  0.0

The max wer value between adv and ori is  2.75

### The demo result of Gaussian defense against CW ASR attack

Try to run the command `python Attack_WER.py --attack CW_ASR_Attack --defense Gaussian` to check the WER result of 2 audio examples with defenses.

### Implemention label Smooth defense against IMP/CW ASR Attacks

Try to run the command `python Attack_WER.py --attack IMP_ASR_Attack --defense Smooth` to check the WER result of 2 audio examples with defenses.

#### If you run the command above correctly, you will get the following results:

The adversarial samples transcripted by Deepspeech is:  HEY SIRI CPOY THE DOOR

The adv label of 0th audio is HEY SIRI CPOY THE DOOR

The adversarial samples transcripted by Deepspeech is:  HEY ALEXA SAVE PACK MY BOX WITH FIVE DOZEN LIQUOR JUGS

The adv label of 1th audio is HEY ALEXA SAVE PACK MY BOX WITH FIVE DOZEN LIQUOR JUGS

The WER between adv txt and target txt is as follow

at_wer_list is [0. 0.]

ao_wer_list is [0.98214286 2.75      ]

The max wer value between adv and target is  0.0

The max wer location between adv and target is  (array([0, 1]),)

The average wer value between adv and target is  0.0

The max wer value between adv and ori is  2.75

### The demo result of Resample defense against CW ASR attack

Try to run the command `python Attack_WER.py --attack IMP_ASR_Attack --defense RS` to check the WER result of 2 audio examples with defenses.

#### If you run the command above correctly, you will get the following results:

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

### The demo result of Resample defense against CW ASR attack

Try to run the command `python Attack_WER.py --attack CW_ASR_Attack --defense RS` to check the WER result of 2 audio examples with defenses.


## ODE-based Defense against IMP ASR attack

As a comparison, you can run the command `python Defense_WER.py` to check the WER result of 2 audio examples with ODE-Based defense. 

#### If you run the command above correctly, you will get the following results:

The groundtruth is :  IN THE HIMALAYAS THE THIRD LARGEST MASS OF ICE AT THE TOP YOU SEE NEW LAKES WHICH A FEW YEARS AGO WERE GLACIERS FORTY PERCENT OF ALL THE PEOPLE IN THE WORLD GET HALF OF THEIR DRINKING WATER FROM THAT MELTING FLOW IN THE ANDES THIS GLACIER IS THE SOURCE OF DRINKING WATER FOR THIS CITY

Without any defense, the adversarial samples transcripted by deepspeech:  HEY SIRI CPOY THE DOOR

The adv target of 0th audio should to be HEY SIRI CPOY THE DOOR

After adding the ODE defense block, the transcription turns to be:  IEHIMALA THE THIRD LARGEST NAS EVICE A THE TOP YOU SEE MELEX  O FEEAR AGORGEE ORI ERCENT OF HALL THE PEOPLE WO IN NHE WORAL GETHALFOF THE RNKN WANTER O MELTING FLO IN THE END IS TISE ER THI SORC OF THREING WONTER FOR THI CITY

The groundtruth is :  WHAT THIS IS DOING

Without any defense, the adversarial samples transcripted by deepspeech:  HEY ALEXA SAVE PACK MY BOX WITH FIVE DOZEN LIQUOR JUGS

The adv target of 1th audio should to be HEY ALEXA SAVE PACK MY BOX WITH FIVE DOZEN LIQUOR JUGS

After adding the ODE defense block, the transcription turns to be:  WHAT DIVE TEME DE INY

at_wer_list is [9. 1.]

ao_wer_list is [0.71428571 1.        ]

After adding Defenses,the max wer value between adv and target is  9.0

After adding Defenses,the max wer location between adv and target is  (array([0]),)

After adding Defenses,the average wer value between adv and target is  5.0

After adding Defenses,the max wer value between adv and ori is  1.0

After adding Defenses,the max wer location between adv and ori is  (array([1]),)

After adding Defenses,the average wer value between adv and ori is  0.8571428571428572




