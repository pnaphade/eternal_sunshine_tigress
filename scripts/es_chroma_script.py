import numpy as np
import matplotlib.pyplot as plt
import librosa as lb
import librosa.display
from audio2numpy import open_audio
import ffmpeg # support for opening mp3 files


# Load in the movie audio
audio, sample_rate = open_audio("/tigress/pnaphade/es_audio.mp3")


# Convert to mono audio, split into halves
audio_mono = lb.core.to_mono(audio.T)
audio_mono_1 = audio_mono[: int(audio_mono.shape[0]/2)]
audio_mono_2 = audio_mono[int(audio_mono.shape[0]/2) :]


# Compute spectrogram of the audio 
chroma_1 = lb.feature.chroma_cqt(audio_mono_1)
chroma_2 = lb.feature.chroma_cqt(audio_mono_2) 


# Take absolute value and convert to dB
chroma_1_db = lb.amplitude_to_db(chroma_1)
chroma_2_db = lb.amplitude_to_db(chroma_2)


# Sliding average to convert spectrogram or chromagram from samples to seconds
def sliding_avg(spect, audio, sample_rate) : 
	
	# Define the window and set the dimensions of the output spectrogram accordingly
	window = np.round(spect.shape[1]/(len(audio)/sample_rate))
	output = np.zeros((spect.shape[0], int(np.round((spect.shape[1]/window)))))
	
	# Index for center of sliding averages
	forward_idx = np.arange(0, spect.shape[1] + 1, window)
	
	for i in range(len(forward_idx)) :
		
		# Make sure the width of the sub-spectrogram is equal to the window
		if spect[:, int(forward_idx[i]) : int(forward_idx[i]) + int(window)].shape[1] != window :
			continue
		
		# Average across the window
		else : 
			output[:, i] = np.mean(spect[:, int(forward_idx[i]) : int(forward_idx[i]) + int(window)], axis = 1).T
	
	return output
	

# Convert chromagrams from samples to seconds
chroma_1_db_smooth = sliding_avg(chroma_1_db, audio_mono_1, sample_rate)
chroma_2_db_smooth = sliding_avg(chroma_2_db, audio_mono_2, sample_rate)


# Splice the chromagrams together
es_chroma  = np.hstack((chroma_1_db_smooth, chroma_2_db_smooth))


# Display the chromagram
fig, ax = plt.subplots()
chroma = lb.display.specshow(es_chroma, x_axis='time', y_axis='chroma', sr=48000, hop_length=48000, ax=ax)
fig.colorbar(chroma, ax=ax)
#plt.show()
