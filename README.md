# ODE-based-Defense-Structure-for-ASR

In this repository, a ODE-based defense structure for end-to-end ASR system is introduced.

## Set Up

**Please install the following necessary packages.**

1, [Adversarial Robustness Toolbox (ART)](https://github.com/Trusted-AI/adversarial-robustness-toolbox) <br>

**Installation:**

ART and its core dependencies can be installed from the PyPI repository using `pip`:<br>

`pip install adversarial-robustness-toolbox`<br>

2, [Deepspeech.pytorch](https://github.com/SeanNaren/deepspeech.pytorch) <br>

Note: Install [PyTorch](https://github.com/pytorch/pytorch#installation) if you haven't already.

**Installation:**

Please follow the [tutorials](https://github.com/SeanNaren/deepspeech.pytorch) provided in this repo to install step by step.

3, [JiWER](https://github.com/jitsi/jiwer)

**Installation:**

`pip install jiwer`

4, [PyTorch Implementation of Differentiable ODE Solvers](https://github.com/rtqichen/torchdiffeq)

**Installation:**

`pip install torchdiffeq`

## Example

### Implemention of CW/IMP attacks against victim model Deepspeech

Try to run the command `python /model/Attack_WER.py` to check the WER result of 2 audio examples without any defenses.

### Implemention of 4 different defenses against CW/IMP ASR Attacks

Try to run the command `python /model/Defense_WER.py` to check the WER result of 2 audio examples with defenses.



