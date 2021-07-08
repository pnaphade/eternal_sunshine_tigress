import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Load in audio data, transpose in preparation for correlation
es_spect = np.load('/tigress/pnaphade/es_spect.npy')
es_spect = es_spect.T

# Load in neural data
masked_data_dir = '/tigress/jamalw/Eternal_Sunshine/scripts/rois/masked_data/'
music_A1_run1  = np.load(masked_data_dir + 'music/a1plus_run1_n12.npy')
music_A1_run2 = np.load(masked_data_dir + 'music/a1plus_run2_n12.npy')
music_rA1_run1  = np.load(masked_data_dir + 'music/rA1_run1_n12.npy')
music_rA1_run2 = np.load(masked_data_dir + 'music/rA1_run2_n12.npy')
no_music_A1_run1  = np.load(masked_data_dir + 'no_music/a1plus_run1_n11.npy')
no_music_A1_run2 = np.load(masked_data_dir + 'no_music/a1plus_run2_n11.npy')
no_music_rA1_run1  = np.load(masked_data_dir + 'no_music/rA1_run1_n11.npy')
no_music_rA1_run2 = np.load(masked_data_dir + 'no_music/rA1_run2_n11.npy')

# Average, concatenate, and transpose neural data in preparation for correlation
def corr_prep(run1_data, run2_data) :
	
	# Make sure we're working with data of the correct type and dimensionality
	if not(isinstance(run1_data, np.ndarray)) or not(isinstance(run2_data, np.ndarray)) :
		raise TypeError("Data must be a numpy ndarray")
		
	if run1_data.ndim != 3 or run2_data.ndim != 3 :
		raise ValueError("Data must be 3 dimensional (voxels x time x subjects)")

	# Average each run across subjects
	run1_data_avg = np.mean(run1_data, axis=2)
	run2_data_avg = np.mean(run2_data, axis=2)

	# Concatenate the data across time
	avg_concat_data = np.concatenate((run1_data_avg, run2_data_avg), axis=1)

	# Transpose and return
	return avg_concat_data.T

music_A1_prepped = corr_prep(music_A1_run1, music_A1_run2)
music_rA1_prepped = corr_prep(music_rA1_run1, music_rA1_run2)
no_music_A1_prepped = corr_prep(no_music_A1_run1, no_music_A1_run2)
no_music_rA1_prepped = corr_prep(no_music_rA1_run1, no_music_rA1_run2)



# Correlate the neural data across time
music_A1_corr = np.corrcoef(music_A1_prepped)
music_rA1_corr = np.corrcoef(music_rA1_prepped)
no_music_A1_corr = np.corrcoef(no_music_A1_prepped)
no_music_rA1_corr = np.corrcoef(no_music_rA1_prepped)

# Figure out which rows in the spectrogram have zero variance (results in nans in correlation)
zero_var_rows = []
for i in range(6452):
	if np.var(es_spect[i, :]) == 0 :
		zero_var_rows.append(i)

#print(zero_var_rows)

# Function for chopping off desired values from a symmetric matrix
def chop(mat, idx_lo, idx_hi) :
	return mat[idx_lo : idx_hi, idx_lo : idx_hi]

# Chop off the appropriate values
es_spect_chop = es_spect[4 : 6431, :] 

# Correlate the chopped audio data across time
audio_corr_chop = np.corrcoef(es_spect_chop)	

# Chop the neural matrices to match shape with the audio RSM
music_A1_corr_chop = chop(music_A1_corr, 4, 6431)
music_rA1_corr_chop = chop(music_rA1_corr, 4, 6431)
no_music_A1_corr_chop = chop(no_music_A1_corr, 4, 6431)
no_music_rA1_corr_chop = chop(no_music_rA1_corr, 4, 6431)

# Group the neural data into a 3D array
neural_mat = np.asarray([music_A1_corr_chop, music_rA1_corr_chop, no_music_A1_corr_chop, no_music_rA1_corr_chop])


# Variables for sliding window correlations
n_rois = 4
n_trs = es_spect_chop.shape[0]
window_width = 30
n_windows = n_trs - window_width
RSM_corrs = np.zeros((n_rois, 2, n_windows))

# Calculate 30 s sliding window correlations between the audio RSM and each neural RSM
for i in range(n_windows) :

	# Isolate the current window for the neural and audio RSMs
	audio_win = chop(audio_corr_chop, i, i+window_width)
	neural_win_mat = np.zeros((n_rois, window_width, window_width))
	for j in range(n_rois) : 
		neural_win_mat[j] = chop(neural_mat[j], i, i+window_width)

	# Pull out the upper triangles of the RSMs for correlation
	triu_idx = np.triu_indices(window_width)
	audio_win_triu = audio_win[triu_idx]
	neural_win_triu_mat = np.zeros((n_rois, audio_win_triu.shape[0]))
	for j in range(n_rois) :
		neural_win_triu_mat[j] = neural_win_mat[j][triu_idx]
		
	# Compute the Pearson correlation and p value
	for j in range(n_rois) :
		RSM_corrs[j, 0, i], RSM_corrs[j, 1, i] = stats.pearsonr(neural_win_triu_mat[j], audio_win_triu)

avg_corrs, avg_pvals = np.mean(RSM_corrs[:, 0, :], axis=1), np.mean(RSM_corrs[:, 1, :], axis=1)

# Print the results
print(f"Neural-Audio Correlations: {np.around(avg_corrs, decimals=5)}")


# Visualize the neural representational similarity matrices
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()
roi_labels = ["Music Bilateral A1", "Music Right A1", "No Music Bilateral A1", "No Music Right A1"]

for i, ax in zip(np.arange(n_rois), axes) :

	im = ax.imshow(neural_mat[i], cmap='jet')
	ax.set_title(roi_labels[i])
	
	if i == 2 or i == 3 :
		ax.set_xlabel("TR")
	
	if i == 0 or i == 2 :
		ax.set_ylabel("TR")

fig.subplots_adjust(right = 0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)

# Visualize the audio representational similarity matrix
fig, ax = plt.subplots()
im = ax.imshow(audio_corr_chop, cmap='jet')
ax.set_xlabel("TR")
ax.set_ylabel("TR")
fig.subplots_adjust(right = 0.8)
cbar_ax = fig.add_axes([0.80, 0.15, 0.05, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax)


# Visualize the neural-audio correlations as a function of time
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

for i, ax in zip(np.arange(n_rois), axes) :

	ax.plot(RSM_corrs[i, 0, :], linewidth=0.5)
	ax.set_title(roi_labels[i])
	ax.set_ylim(0, 0.8)
	
	if i == 2 or i == 3 :
		ax.set_xlabel("TR")
	
	if i == 0 or i == 2 :
		ax.set_ylabel("Neural-Audio Correlation")

#plt.show()
