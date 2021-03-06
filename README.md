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

### Implemention of [CW ASR Attack](https://arxiv.org/abs/1801.01944) against ASR System DeepSpeech.

Operate like above, try to run the command ` python model/Attack_WER.py --attack CW_ASR_Attack` to check the WER result of 2 audio examples without any defenses. 

### Implemention of [PGD ASR Attack](https://arxiv.org/abs/1906.03333) against ASR System DeepSpeech.

Operate like above, try to run the command ` python model/PGD_Attack_WER.py --attack PGD_ASR_Attack` to check the WER result of 2 audio examples without any defenses. 

#### Please refer to the results in the [Result_of_Attacks](https://github.com/Winterzhangwinter/ODE-Based-Defense-Structure-for-ASR-System/blob/main/Result_of_Attacks.txt) to determine if you are running the commands above correctly.

## Voice Defenses against IMP ASR Attack.

On the other hand, you can add commonly used defense methods against the adeversarial examples.

### The Demo Result of Gaussian Defense against IMP ASR Attack

Try to run the command `python model/Attack_WER.py --attack IMP_ASR_Attack --defense Gaussian` to check the WER result of 2 audio examples with defenses.

### The Demo Result of Smooth Defense against IMP ASR Attack

Try to run the command `python model/Attack_WER.py --attack IMP_ASR_Attack --defense Smooth` to check the WER result of 2 audio examples with defenses. 

### The Demo Result of Resample Defense against IMP ASR Attack

Try to run the command `python model/Attack_WER.py --attack IMP_ASR_Attack --defense Resample` to check the WER result of 2 audio examples with defenses.

### The Demo Result of ODE-based Defense against IMP ASR Attack

Try to run the command `python model/Defense_WER.py` to check the WER result of 2 audio examples with defenses.

#### Please refer to the results in the [Result_IMP_Defense](https://github.com/Winterzhangwinter/ODE-Based-Defense-Structure-for-ASR-System/blob/main/Result_IMP_Defense.txt) to determine if you are running the commands above correctly.

## Voice Defenses against C&W ASR Attack.

### The Demo Result of Gaussian Defense against C&W ASR attack

Try to run the command `python model/Attack_WER.py --attack CW_ASR_Attack --defense Gaussian` to check the WER result of 2 audio examples with defenses. 

### The Demo Result of Lable Smooth Defense against C&W ASR Attack

Try to run the command `python model/Attack_WER.py --attack CW_ASR_Attack --defense Smooth` to check the WER result of 2 audio examples with defenses.

### The Demo Result of Resample Defense against C&W ASR attack

Try to run the command `python model/Attack_WER.py --attack CW_ASR_Attack --defense Resample` to check the WER result of 2 audio examples with defenses.

### The Demo Result of ODE-based Defense against C&W ASR Attack

Try to run the command `python model/CW_Defense_WER.py` to check the WER result of 2 audio examples with defenses.

#### Please refer to the results in the [Result_CW_Defense](https://github.com/Winterzhangwinter/ODE-Based-Defense-Structure-for-ASR-System/blob/main/Result_CW_Defense.txt) to determine if you are running the commands above correctly.

## Voice Defenses against PGD ASR Attack.

### The Demo Result of Gaussian Defense against PGD ASR attack

Try to run the command `python model/PGD_Attack_WER.py --attack PGD_ASR_Attack --defense Gaussian` to check the WER result of 2 audio examples with defenses. 

### The Demo Result of Lable Smooth Defense against PGD ASR attack

Try to run the command `python model/PGD_Attack_WER.py --attack PGD_ASR_Attack --defense Smooth` to check the WER result of 2 audio examples with defenses. 

### The Demo Result of Resample Defense against PGD ASR attack

Try to run the command `python model/PGD_Attack_WER.py --attack PGD_ASR_Attack --defense Resample` to check the WER result of 2 audio examples with defenses. 

### The Demo Result of ODE-based Defense against PGD ASR Attack

Try to run the command `python model/PGD_ODE.py` to check the WER result of 2 audio examples with defenses.

#### Please refer to the results in the [Result_PGD_Defense](https://github.com/Winterzhangwinter/ODE-Based-Defense-Structure-for-ASR-System/blob/main/Result_PGD_Defense.txt) to determine if you are running the commands above correctly.
