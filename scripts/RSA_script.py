import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from RSA_functions import RSA, corr_prep
import re
from pathlib import Path
import os
import glob


# Load in the audio features, transpose in preparation for correlation
feat_dir = "/tigress/pnaphade/Eternal_Sunshine/results/RSA"
feat_paths = glob.glob(os.path.join(feat_dir, "es*"))
features = [np.load(path).T for path in feat_paths]

# Pull out the labels of each feature from their filepaths using regex
feat_labels = [re.search('es_(.+?).npy', path).group(1) for path in feat_paths]

# Load in neural data
masked_dir = "/tigress/pnaphade/Eternal_Sunshine/scripts/rois/masked_data/"

A1_data = ["music/a1plus_run1_n12.npy", "music/a1plus_run2_n12.npy", "music/rA1_run1_n12.npy", "music/rA1_run2_n12.npy","no_music/a1plus_run1_n11.npy","no_music/a1plus_run2_n11.npy","no_music/rA1_run1_n11.npy","no_music/rA1_run2_n11.npy"]

control_data = ["music/brainstem_run1_n12.npy", "music/brainstem_run2_n12.npy", "music/occipital_pole_run1_n12.npy", "music/occipital_pole_run2_n12.npy", "no_music/brainstem_run1_n11.npy", "no_music/brainstem_run2_n11.npy", "no_music/occipital_pole_run1_n11.npy", "no_music/occipital_pole_run2_n11.npy"]

# Choose which dataset to analyze
neural_runs = [np.load(masked_dir + run) for run in control_data]

# Labels for displaying results
#corr_labels = ["Music A1", "Music rA1", "No Music A1", "No Music rA1"]
corr_labels = ["Music Brainstem", "Music Occipital Pole", "No Music Brainstem", "No Music Occipital Pole"]




# Prepare the neural data for correlation
neural_prepped = []
for i in  np.arange(int(len(neural_runs)/2)) :
	neural_prepped.append(corr_prep(neural_runs[2*i], neural_runs[2*i+1]))

# Figure out which rows in the features have zero variance (results in nans in correlation)
n_rows = features[0].shape[0]

# 2D list giving the zero variance rows in each feature
rows = [[i for i in range(n_rows) if np.var(feature[i, :]) == 0] for feature in features]

# For comparison, set the dictionary's values as the number zero variance rows in each feature
n_zero_var_rows = [len(rows[i]) for i in np.arange(len(rows))]
zero_var_rows = dict(zip(feat_labels, n_zero_var_rows))

# Chop off the appropriate values in the features and neural data for consistency
for i in np.arange(len(neural_prepped)) : 
	features[i] = features[i][4:6431, :] 
	neural_prepped[i] = neural_prepped[i][4:6431, :]		



# Perform the RSA

# Data parameters
n_feats = len(features)
n_neurdata = len(neural_prepped)
n_RSMs = n_neurdata + 1
n_trs = neural_prepped[0].shape[0]

# Data storage
RSMs = np.zeros((n_feats, n_RSMs, n_trs, n_trs))
corrs = np.zeros((n_feats, n_neurdata))
sliding_corrs = np.zeros((n_feats, n_neurdata, n_trs-30)) #

# Loop over each audio feature
for i, feature in enumerate(features) :
	
	# Skip over i = 1 because of the many zero variance rows in mel-scaled spectrogram
	if i == 1 :
		continue

	print(f"Performing RSA using audio feature {i+1} of {n_feats}")
	
	#slide_corrs_temp = []
	
	# Loop over each neural dataset
	for j, neurdata in enumerate(neural_prepped) :
		
		results = RSA(neurdata, feature, sliding_window=True, window_width=30) 
		
		# Record the current neural RSM
		RSMs[i, j] = results[0]

		# Record the correlations between the current two RSMs
		corrs[i, j] = results[2]

		# Record the sliding correlations betweeen the two current RSMs
		sliding_corrs[i, j] = results[4]
		#slide_corrs_temp.append(results[4])
		
	# Record the audio feature RSM
	RSMs[i, 4] = results[1]

	# Store the sliding correlations as ndarrays 
	#slide_corrs_temp = np.asarray(slide_corrs_temp)


# Print the results
print("\nResults")
print("-------\n")

for i, feature in enumerate(features) :
	
	# Skip over the mel-scaled spectrogram	
	if i == 1 :
		continue
	
	print(f"{feat_labels[i]}")

	for j in np.arange(n_neurdata) :
		print(f"{corr_labels[j]}-Audio correlation: {np.around(corrs[i, j], decimals=4)}")

	if not(i == len(features)-1) :
		print("\n", end='')
'''

# Save the between RSM correlations and RSMs
save_dir = "/tigress/pnaphade/Eternal_Sunshine/results/RSA/"
roi = "A1"
corrs_path = Path(save_dir + roi + "_corrs.npy")
RSMs_path = Path(save_dir + roi + "_RSMs.npy")
	
if not(corrs_path.exists()) :
	np.save(corrs_path, corrs)

if not(RSMs_path.exists()) :
	np.save(RSMs_path, RSMs)


# Plotting
roi_labels = ["Music Bilateral A1", "Music Right A1", "No Music Bilateral A1", "No Music Right A1"]

# Visualize the neural representational similarity matrices
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()
for i, ax in zip(np.arange(n_neurdata), axes) :

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

for i, ax in zip(np.arange(n_neurdata), axes) :

	ax.plot(sliding_corrs[i], linewidth=0.5)
	ax.set_title(roi_labels[i])
	ax.set_ylim(0, 0.8)
	ax.text(5050, 0.72, r'$r_{mean}$' + f" = {np.around(corrs, decimals=4)[i]}")
		
	if i == 2 or i == 3 :
		ax.set_xlabel("TR")
	
	if i == 0 or i == 2 :
		ax.set_ylabel("Neural-Audio Correlation")

'''
'''
# Correlations for final beach house scene
RSM_corrs_beachhouse = RSM_corrs[:, 0, 5410:5470]
avg_corrs_beachhouse = np.mean(RSM_corrs_beachhouse, axis=1)
print(f"Neural-Audio correlations, final beach house scene: {avg_corrs_beachhouse}")

# Visualize beach house scene correlations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

for i, ax in zip(np.arange(n_neurdata), axes) :

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
