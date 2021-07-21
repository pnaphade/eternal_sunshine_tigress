import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from RSA_functions import RSA, corr_prep

# Load in audio data, transpose in preparation for correlatio
# Choose the chromagram this time
es_spect = np.load("/tigress/pnaphade/results/es_chroma.npy")
es_spect = es_spect.T

# Load in neural data
masked_data_dir = "/tigress/jamalw/Eternal_Sunshine/scripts/rois/masked_data/"
dir_endings = ["music/a1plus_run1_n12.npy", "music/a1plus_run2_n12.npy", "music/rA1_run1_n12.npy", "music/rA1_run2_n12.npy","no_music/a1plus_run1_n11.npy","no_music/a1plus_run2_n11.npy","no_music/rA1_run1_n11.npy","no_music/rA1_run2_n11.npy"]

neural_runs = []

for ending in dir_endings :
	neural_runs.append(np.load(masked_data_dir + ending))
	


# Prepare the neural data for correlation
neural_prepped = []

for run, i in zip(neural_runs, np.arange(4)) :
	neural_prepped.append(corr_prep(neural_runs[2*i], neural_runs[2*i+1]))

# Figure out which columns in the spectrogram have zero variance (results in nans in correlation)
zero_var_cols = []
for i in range(neural_prepped[0].shape[0]):
	if np.var(es_spect[i, :]) == 0 :
		zero_var_cols.append(i)

# Chop off the appropriate values
es_spect_chop = es_spect[4:6431, :] 

for dataset, i in zip(neural_prepped, np.arange(4)) : 
	neural_prepped[i] = dataset[4:6431, :]		



# Perform the RSA
corrs = np.zeros(4)
sliding_corrs = []
RSMs = np.zeros((5, neural_prepped[0].shape[0], neural_prepped[0].shape[0]))
for area, i in zip(neural_prepped, np.arange(4)) :
	
	results = RSA(area, es_spect_chop, sliding_window=True, window_width=30) 
	
	# Record the correlations between the current two RSMs
	corrs[i] = results[2]

	# Record the sliding correlations betweeen the two current RSMs
	sliding_corrs.append(results[4])

	# Record the current neural RSM
	RSMs[i] = results[0]
	
# Record the audio RSM
RSMs[4] = results[1]

# Convert sliding correlations into an ndarray
sliding_corrs = np.asarray(sliding_corrs)



# Plotting
roi_labels = ["Music Bilateral A1", "Music Right A1", "No Music Bilateral A1", "No Music Right A1"]

# Visualize the neural representational similarity matrices
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()
for i, ax in zip(np.arange(4), axes) :

	im = ax.imshow(RSMs[i], cmap='jet')
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
im = ax.imshow(RSMs[4], cmap='jet')
ax.set_xlabel("TR")
ax.set_ylabel("TR")
fig.subplots_adjust(right = 0.8)
cbar_ax = fig.add_axes([0.80, 0.15, 0.05, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax)


# Visualize the neural-audio correlations as a function of time
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

for i, ax in zip(np.arange(4), axes) :

	ax.plot(sliding_corrs[i], linewidth=0.5)
	ax.set_title(roi_labels[i])
	ax.set_ylim(0, 0.8)
	ax.text(5050, 0.72, r'$r_{mean}$' + f" = {np.around(corrs, decimals=4)[i]}")
		
	if i == 2 or i == 3 :
		ax.set_xlabel("TR")
	
	if i == 0 or i == 2 :
		ax.set_ylabel("Neural-Audio Correlation")


'''
# Correlations for final beach house scene
RSM_corrs_beachhouse = RSM_corrs[:, 0, 5410:5470]
avg_corrs_beachhouse = np.mean(RSM_corrs_beachhouse, axis=1)
print(f"Neural-Audio correlations, final beach house scene: {avg_corrs_beachhouse}")

# Visualize beach house scene correlations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

for i, ax in zip(np.arange(n_rois), axes) :

	ax.plot(RSM_corrs_beachhouse[i, :], linewidth=1)
	ax.set_title(roi_labels[i])
	ax.set_ylim(0, 1)
	ax.text(45, 0.9, r'$r_{mean}$' + f" = {np.around(avg_corrs_beachhouse, decimals=4)[i]}")
		
	if i == 2 or i == 3 :
		ax.set_xlabel("TR")
	
	if i == 0 or i == 2 :
		ax.set_ylabel("Neural-Audio Correlation")

#plt.show()
'''
