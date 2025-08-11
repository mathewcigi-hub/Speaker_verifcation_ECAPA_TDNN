
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier

classifier = EncoderClassifier.from_hparams(source="/home/batch3/Math/speechbrain/recipes/VoxCeleb/SpeakerRec/For_Embedding")
#classifier = EncoderClassifier.from_hparams(source="/home/batch3/Math/speechbrain/recipes/VoxCeleb/SpeakerRec/pretrained_models_somthing/EncoderClassifier-8f6f7fdaa9628acf73e21ad1f99d5f83")
signal, fs =torchaudio.load('/home/batch3/Math/speechbrain/tests/samples/ASR/spk1_snt1.wav')
embeddings = classifier.encode_batch(signal)
print(embeddings)


'''


import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
signal, fs =torchaudio.load('/home/batch3/Math/speechbrain/tests/samples/ASR/spk1_snt1.wav')
embeddings = classifier.encode_batch(signal)
print(embeddings)

'''



