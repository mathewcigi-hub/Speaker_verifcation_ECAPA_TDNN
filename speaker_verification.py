
import torchaudio
from speechbrain.inference.speaker import SpeakerRecognition

source_dir = "/home/batch3/Math/speechbrain/recipes/VoxCeleb/SpeakerRec/For_Embedding"
pretrained_model_dir = "/home/batch3/Math/speechbrain/recipes/VoxCeleb/SpeakerRec/pretrained_models/EncoderClassifier-ddd2b5fa6fa239c1beddced8a394ae2a"

#verification = SpeakerRecognition.from_hparams(source=source_dir, savedir=pretrained_model_dir)
verification = SpeakerRecognition.from_hparams(source=source_dir)

same_speaker_1 = "/home/batch3/Math/speechbrain/tests/samples/ASR/spk1_snt2.wav"
same_speaker_2 = "/home/batch3/Math/speechbrain/tests/samples/ASR/spk1_snt5.wav"
different_speaker_1 = "/home/batch3/Math/speechbrain/tests/samples/ASR/spk1_snt1.wav"
different_speaker_2 = "/home/batch3/Math/speechbrain/tests/samples/ASR/spk2_snt6.wav"

score, prediction = verification.verify_files(same_speaker_1, same_speaker_2)
print("\n\t Score: {:.4f}, Prediction: {}".format(score.item(), bool(prediction.item())))

score, prediction = verification.verify_files(different_speaker_1, different_speaker_2)
print("\n\t Score: {:.4f}, Prediction: {}\n".format(score.item(), bool(prediction.item())))



'''
from speechbrain.inference.speaker import SpeakerRecognition
verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")
score, prediction = verification.verify_files("tests/samples/ASR/spk1_snt1.wav", "tests/samples/ASR/spk2_snt1.wav") # Different Speakers
score, prediction = verification.verify_files("tests/samples/ASR/spk1_snt1.wav", "tests/samples/ASR/spk1_snt2.wav") # Same Speaker

'''



