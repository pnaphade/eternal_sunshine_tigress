import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from RSA_functions import RSA, corr_prep
import re
from pathlib import Path
import os
import glob
import xlrd


#Load in the audio features, transpose in preparation for correlation
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
neural_runs = [np.load(masked_dir + run) for run in A1_data]
#neural_runs = []

#for i in np.arange(8) :
	#rand_data = np.random.default_rng().uniform(-2, 2, (1500, 3240, 12))
	#neural_runs.append(rand_data)

# Labels for displaying results
corr_labels = ["Music A1", "Music rA1", "No Music A1", "No Music rA1"]
#corr_labels = ["Music Brainstem", "Music Occipital Pole", "No Music Brainstem", "No Music Occipital Pole"]
#corr_labels = ["random 1", "random 2", "random 3", "random 4"]



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
	
	print(f"Performing RSA using audio feature {i+1} of {n_feats}")
	
	# Loop over each neural dataset
	for j, neurdata in enumerate(neural_prepped) :
		
		results = RSA(neurdata, feature, sliding_window=True, window_width=30) 
		
		# Record the current neural RSM
		RSMs[i, j] = results[0]

		# Record the correlations between the current two RSMs
		corrs[i, j] = results[2]

		# Record the sliding correlations betweeen the two current RSMs
		sliding_corrs[i, j] = results[4]
			
	# Record the audio feature RSM
	RSMs[i, 4] = results[1]

	# Store the sliding correlations as ndarrays 
	#slide_corrs_temp = np.asarray(slide_corrs_temp)


# Credits scene regressor
# The credits are the last 205 seconds of the movie. Extract the last 205-30 = 175
# seconds of the movie because of the 30s sliding windows
credits_sliding_corrs = sliding_corrs[:, :, -175:]
credits_corrs = np.mean(credits_sliding_corrs, axis=2)

master_corrs = [corrs, credits_corrs]

# Print the results
print("\nResults")
print("-------\n")

for h, corrs in enumerate(master_corrs) :

	if h == 0 :
		print("Whole-Movie Correlations")
	else :
		print("Credits Scene Correlations")

	for i, feature in enumerate(features) :
		
		print(f"{feat_labels[i]}")

		for j in np.arange(n_neurdata) :
			print(f"{corr_labels[j]}-Audio correlation: {np.around(corrs[i, j], decimals=4)}")

		if not(i == 3 and h == 1) :
			print("\n", end='')


'''
# Read the Scene Song Notations workbook
music_times_path = "/tigress/pnaphade/Eternal_Sunshine/data/scene_song_notations.xlsx"
music_times_wb = xlrd.open_workbook(music_times_path)
sheet = music_times_wb.sheet_by_index(0)

# Extract the onset and offset columns
onset_cells = sheet.col(1)
offset_cells = sheet.col(2)

# Pop the headers of the columns
onset_cells.pop(0)
offset_cells.pop(0)

# Extract the values of the columns
onset_vals = [cell.value for cell in onset_cells]
offset_vals = [cell.value for cell in offset_cells]

# Function for converting to seconds
def to_seconds(time):
	
	hours, minutes, seconds = time.split(':')
	
	total_seconds = int(hours)*3600 + int(minutes)*60 + int(seconds)
	
	return total_seconds


# Convert the times into seconds
onset_sec = np.asarray([to_seconds(val) for val in onset_vals])
offset_sec = np.asarray([to_seconds(val) for val in offset_vals])

# Adjust 4 seconds forward due to beginning silence
onset_sec, offset_sec = onset_sec-4, offset_sec-4
'''

'''
# plot the credits results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes = axes.flatten()

# Skip the mel-scaled spectrogram
for i, ax in zip(np.asarray([0, 2, 3]), axes) :
	
	# plot the music A1 data
	ax.plot(credits_sliding_corrs[i, 0], 'r-', linewidth=1, label=corr_labels[0])

	# plot the no music A1 data
	ax.plot(credits_sliding_corrs[i, 2], 'b-', linewidth=1, label=corr_labels[2])
	
	ax.set_title(feat_labels[i])
		
	ax.set_xlabel("TR")
	
	ax.set_ylim(0.2, 0.7)	
	ax.legend(loc = 'upper left')
	
	if i == 0 :
		ax.set_ylabel("Neural-Audio Correlation")
'''
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
