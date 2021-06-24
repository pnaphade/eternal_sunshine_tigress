import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# File path for audio data
es_spect = np.load('/tigress/pnaphade/es_spect.npy')
'''
# File path for neural data
masked_data_dir = '/tigress/jamalw/Eternal_Sunshine/scripts/rois/masked_data/'

# Music Bilateral A1 data
music_A1_run1  = np.load(masked_data_dir + 'music/a1plus_run1_n12.npy')
music_A1_run2 = np.load(masked_data_dir + 'music/a1plus_run2_n12.npy')

# Music Right A1 data
music_rA1_run1  = np.load(masked_data_dir + 'music/rA1_run1_n12.npy')
music_rA1_run2 = np.load(masked_data_dir + 'music/rA1_run2_n12.npy')

# No Music Bilateral A1 data
no_music_A1_run1  = np.load(masked_data_dir + 'no_music/a1plus_run1_n11.npy')
no_music_A1_run2 = np.load(masked_data_dir + 'no_music/a1plus_run2_n11.npy')

# No Music Right A1 data
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
	avg_concat_data  = np.concatenate((run1_data_avg, run2_data_avg), axis=1)

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
'''

# Correlate the audio data across time
audio_corr = np.corrcoef(es_spect.T)

audio_corr_num = audio_corr[~np.isnan(audio_corr)]
nan_idx_x = np.asarray(np.where(np.isnan(audio_corr))[0])
nan_idx_y = np.asarray(np.where(np.isnan(audio_corr))[1])

#fig = plt.figure(figsize=(15, 5))
fig, ax = plt.subplots()
ax.plot(np.asarray(nan_idx_x), np.asarray(nan_idx_y), 'bo', markersize=1)
plt.show()

'''
# Visualize the neural representational similarity matrices
fig1 = plt.figure(1, figsize=(10, 10))

ax = plt.subplot(2, 2, 1)
im = ax.imshow(music_A1_corr, cmap='jet')
ax.set_title("Music Bilateral A1 RSM")
ax.set_xlabel("TR")
ax.set_ylabel("TR")

ax = plt.subplot(2, 2, 2)
im = ax.imshow(music_rA1_corr, cmap='jet')
ax.set_title("Music Right A1 RSM")
ax.set_xlabel("TR")
ax.set_ylabel("TR")

ax = plt.subplot(2, 2, 3)
im = ax.imshow(no_music_A1_corr, cmap='jet')
ax.set_title("No Music Bilateral A1 RSM")
ax.set_xlabel("TR")
ax.set_ylabel("TR")

ax = plt.subplot(2, 2, 4)
im = ax.imshow(no_music_rA1_corr, cmap='jet')
ax.set_title("No Music Right A1 RSM")
ax.set_xlabel("TR")
ax.set_ylabel("TR")

# Colorbar
fig1.subplots_adjust(right = 0.8)
cbar_ax = fig1.add_axes([0.85, 0.15, 0.05, 0.7])
fig1.colorbar(im, cax=cbar_ax)

# Visualize the audio representational similarity matrix
fig2 = plt.figure(2)
ax = plt.subplot()
im = ax.imshow(audio_corr, cmap='jet')
ax.set_title("Audio RSM")
ax.set_xlabel("TR")
ax.set_ylabel("TR")
fig2.subplots_adjust(right = 0.8)
cbar_ax = fig2.add_axes([0.85, 0.15, 0.05, 0.7])
fig2.colorbar(im, cax=cbar_ax)
plt.show()


# Isolate the upper triangles of the RSMs for correlation
triu_idx = np.triu_indices(6452)
audio_triu = audio_corr[triu_idx]
neural_mat = np.asarray([music_A1_corr, music_rA1_corr, no_music_A1_corr, no_music_rA1_corr])
neural_triu_mat = np.zeros((4, audio_triu.shape[0]))
for i in range(4) :
	neural_triu_mat[i]  = neural_mat[i][triu_idx]


# Calculate the correlations between the audio RSM and each neural RSM
roi_labels = ["Music Bilateral A1", "Music Right A1", "No Music Bilateral A1", "No Music Right A1"]
RSM_corrs = np.zeros((2, 4))
for i in range(4) :

	# Compute Pearson correlation
	RSM_corrs[0, i] = stats.pearsonr(neural_triu_mat[i], audio_triu)[0]

	# Compute Spearman correlation
	RSM_corrs[1, i] = stats.spearmanr(neural_triu_mat[i], audio_triu)[0]
'''
