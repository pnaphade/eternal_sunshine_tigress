import numpy as np
import librosa as lb
from audio2numpy import open_audio
import ffmpeg # support for opening mp3 files

def extract(feature_type, audio_filepath) :

	"""
	
	Extract a feature from an audio file using librosa.

	Parameters
	---------
	feature_type (str) : 
		The feature to be extracted. Valid types are "spect", "mel_spect", "chroma_cqt",
		and "mfcc" for extracting a spectrogram, a mel-scaled spectrogram, a contant-Q 
		chromagram, or the mel-frequency cepstral coefficients, respectively.

	audio_filepath (str) :
		The location of the audio file.

	Returns 
	-------
	extracted_feature (numpy.ndarray) :
		The representation of the desired feature across time.
	
	""" 
	
	# Map feature_type to to the appropriate librosa function
	feature_functions = {
			"spect" : lb.stft, 
			"mel_spect" : lb.feature.melspectrogram, 
			"chroma_cqt" : lb.feature.chroma_cqt,
			"mfcc" : lb.feature.mfcc,
			}

	# Handle type, value errors
	if not(feature_type in list(feature_functions.keys())) :
		raise ValueError("feature_type must be \"spect\", \"chroma_cqt\", or \"mfcc\"")
	if not((isinstance(audio_filepath, str))) :
		raise TypeError("audio_filepath must be a string")
	
	# Load in the audio
	audio, sample_rate = open_audio(audio_filepath)

	# Convert to mono audio, transpose if necessary so that lb.to_mono gets input of shape (2, n)
	if audio.shape[1] == 2 :
		audio = audio.T
	audio_mono = lb.to_mono(audio)


	# Split the audio in half for computational ease
	audio_1 = audio_mono[: int(audio_mono.shape[0]/2)]
	audio_2 = audio_mono[int(audio_mono.shape[0]/2) :]

	
	# Extract the desired feature, noting that lb.stft doesn't take sample rate as an argument
	if feature_type == "spect" :
		feat_raw_1 = feature_functions[feature_type](audio_1)
		feat_raw_2 = feature_functions[feature_type](audio_2)
	else :
		feat_raw_1 = feature_functions[feature_type](audio_1, sr=sample_rate)
		feat_raw_2 = feature_functions[feature_type](audio_2, sr=sample_rate)


	# Perform the appropriate conversions
	if feature_type == "spect" :
		feat_1 = lb.amplitude_to_db(np.abs(feat_raw_1))
		feat_2 = lb.amplitude_to_db(np.abs(feat_raw_2))

	if feature_type = "mel_spect" :
		feat_1 = lb.power_to_db(feat_raw_1)
		feat_2 = lb.power_to_db(feat_raw_2)

	if feature_type == "chroma_cqt" :
		feat_1 = lb.amplitude_to_db(feat_raw_1)
		feat_2 = lb.amplitude_to_db(feat_raw_2)
	
	# No conversions necessary for mfccs
	if feature_type == "mfcc" :
		feat_1 = feat_raw_1
		feat_2 = feat_raw_2

	# Convert to seconds
	feat_smooth_1 = sliding_avg(feat_1, audio_1, sample_rate)
	feat_smooth_2 = sliding_avg(feat_2, audio_2, sample_rate)
	
	# Splice the feature representations together
	extracted_feature = np.hstack((feat_smooth_1, feat_smooth_2))

	return extracted_feature



def sliding_avg(feat, audio, sample_rate) : 

	"""
	
	Convert representation of audio feature (spectrogram, chromagram, etc) from samples
	to seconds using sliding averages.

	Parameters
	----------
	feat (numpy.ndarray) :
		The audio feature extracted from librosa. If the feature is complex, its
		absolute value must be taken before calling this function.

	audio (numpy.ndarray) :
		The audio that was used to extract the feature. Must be mono audio (one
		dimensional array)

	sample_rate (int) : 
		The sample rate of the audio.

	Returns
	-------
	sliding_feat (numpy.ndarray) :
		The audio feature converted to seconds.

	"""	
	
	# Handle type, value errors
	if not(isinstance(feat, np.ndarray)) or not(isinstance(audio, np.ndarray)) :
		raise TypeError("feat and audio must be numpy ndarrays")
	if not(isinstance(sample_rate, int)) : 
		raise TypeError("sample_rate must be an integer")
	if (np.iscomplexobj(feat)) :
		raise ValueError("feat cannot be complex")
	if not(audio.ndim == 1) :
		raise ValueError("audio must be one dimensional")
	feat
	# Define the window and set the dimensions of the sliding average feature accordingly
	window = np.round(feat.shape[1]/(len(audio)/sample_rate))
	sliding_feat = np.zeros((feat.shape[0], int(np.round((feat.shape[1]/window)))))
	
	# Index for center of sliding averages
	forward_idx = np.arange(0, feat.shape[1] + 1, window)
	
	for i in range(len(forward_idx)) :
		
		# Make sure the width of the sub-featrogram is equal to the window
		if feat[:, int(forward_idx[i]) : int(forward_idx[i]) + int(window)].shape[1] != window :
			continue
		
		# Average across the window
		else : 
			sliding_feat[:, i] = np.mean(feat[:, int(forward_idx[i]) : int(forward_idx[i]) + int(window)], axis = 1).T
	
	return sliding_feat
	
