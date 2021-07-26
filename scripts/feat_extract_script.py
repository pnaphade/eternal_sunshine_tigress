import numpy as np
import matplotlib.pyplot as plt
import librosa as lb
import librosa.display
import feat_extract_functions as fex
import pathlib


# Load in the movie audio
audio_filepath = "/tigress/pnaphade/Eternal_Sunshine/data/es_audio.mp3"


# The features we can extract
features = ["spect", "mel_spect", "chroma_cqt", "mfcc"]
n_feats = len(features)


# Extract the features 
es_features = []

for feature in features :
	print(f"Computing {feature}")
	es_features.append(fex.extract(feature, audio_filepath))


# Save the features
save_dir = "/tigress/pnaphade/Eternal_Sunshine/results/RSA/"
for feature, i in zip(es_features, np.arange(n_feats)) :
	feat_path = pathlib.Path(save_dir + "es_" + features[i] + ".npy")
	if not(feat_path.exists()) :
		np.save(feat_path, feature)


# Display the features
feature_labels = ["Spectrogram", "Mel-Scaled Spectrogram", "Constant-Q Chromagram", "Mel-Frequency Cepstral Coefficients"]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
for ax, feature, i in zip(axes, es_features, np.arange(n_feats)) :
	
	# Use chroma as the y axis for the constant-q chromagram
	if i == 2 :
		image = lb.display.specshow(feature, x_axis='time', y_axis='chroma', sr=48000, hop_length=48000, ax=ax)
		fig.colorbar(image, ax=ax, format="%+2.f dB")
	
	# No decibels for mfccs
	elif i == 3 :
		image = lb.display.specshow(feature, x_axis='time', sr=48000, hop_length=48000, ax=ax)
		fig.colorbar(image, ax=ax)
	
	else: 
		image = lb.display.specshow(feature, x_axis='time', y_axis='hz', sr=48000, hop_length=48000, ax=ax)
		fig.colorbar(image, ax=ax, format="%+2.f dB")
	
	ax.set_title(feature_labels[i])	

#plt.show()
