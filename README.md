# Speaker_verifcation_ECAPA_TDNN

A deep learning project for speaker recognition using the [SpeechBrain](https://speechbrain.github.io/) toolkit and the [VoxCeleb1](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/) dataset. The model is based on the ECAPA-TDNN architecture for state-of-the-art performance in speaker verification tasks.



## Overview
This project implements a **speaker recognition system** using the **ECAPA-TDNN** architecture from the SpeechBrain toolkit.  
It uses the VoxCeleb1 dataset for training and evaluation.  
The model extracts speaker embeddings and verifies whether two audio samples belong to the same speaker.  
Applications include authentication systems, voice-based security, and speaker diarization.


## Dataset
We use the [VoxCeleb1](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/) dataset:
- 1,251 speakers
- Over 100,000 utterances
- Audio collected from YouTube interviews

**Preprocessing:**
- Resampled to 16 kHz
- Converted to mono
- Silence removed using `torchaudio`'s VAD



## Model
The **ECAPA-TDNN** (Emphasized Channel Attention, Propagation and Aggregation Time Delay Neural Network) is designed for speaker embedding extraction.

Key features:
- **Squeeze-Excitation blocks** for channel attention
- **Res2Net** modules for multi-scale feature extraction
- Aggregation layers for robust embeddings
- Trained with Additive Margin Softmax loss

Implementation is based on [SpeechBrain's ECAPA-TDNN recipe](https://speechbrain.github.io/recipes/VoxCeleb/SpeakerRec.html).


## Installation
Clone the repository and install dependencies:
git clone https://github.com/yourusername/speaker-recognition-ecapa.git
cd speaker-recognition-ecapa
pip install -r requirements.txt


## Results
| Metric       | Value  |
|--------------|--------|
| EER (%)      | 2.52%  |
| MinDCF       | 0.269  |

Model achieves competitive performance compared to state-of-the-art.

## References
- ECAPA-TDNN.ppt shows about the result and alos explain about the code sections
- https://arxiv.org/pdf/2005.07143 (ECAPA-TDNN reference paper)

