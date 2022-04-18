import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from RSA_functions import RSA, corr_prep
import re
from pathlib import Path
import glob
import xlrd


# Load in the audio features, transpose in preparation for correlation
feat_dir = "/tigress/pnaphade/Eternal_Sunshine/results/RSA/"

# Load in hrf-convolved audio features, pull out labels
##### These audio features are just for the "Main Title" song (from youtube, no other background noise) #####
feat_paths = glob.glob(feat_dir + "hrf_main*")
features = [np.load(path).T for path in feat_paths]
feat_labels = [re.search('hrf_main_title_(.+?).npy', path).group(1) for path in feat_paths]

# Load in neural data
masked_dir = "/tigress/pnaphade/Eternal_Sunshine/scripts/rois/masked_data/"

A1_data = ["music/a1plus_run1_smooth_n25.npy", "music/a1plus_run2_smooth_n25.npy", "no_music/a1plus_run1_smooth_n25.npy","no_music/a1plus_run2_smooth_n25.npy"]

rA1_data = ["music/rA1_run1_n12.npy", "music/rA1_run2_n12.npy","no_music/rA1_run1_n11.npy", "no_music/rA1_run2_n11.npy"]

brainstem_data = ["music/brainstem_run1_n12.npy", "music/brainstem_run2_n12.npy", "no_music/brainstem_run1_n11.npy", "no_music/brainstem_run2_n11.npy"]

occipital_data = ["music/occipital_pole_run1_n12.npy", "music/occipital_pole_run2_n12.npy", "no_music/occipital_pole_run1_n11.npy", "no_music/occipital_pole_run2_n11.npy"]

dmn_data = ["music/dmnA_run1_n12.npy", "music/dmnA_run2_n12.npy", "no_music/dmnA_run1_n11.npy", "no_music/dmnA_run2_n11.npy"]

# Choose which dataset to analyze
data_choice = "rA1"

if data_choice == "A1" :
        neural_runs = [np.load(masked_dir + run) for run in A1_data]
        corr_labels = ["Music A1", "No Music A1"]

if data_choice == "rA1" :
        neural_runs = [np.load(masked_dir + run) for run in rA1_data]
        corr_labels = ["Music rA1", "No Music rA1"]

if data_choice == "brainstem" :
        neural_runs = [np.load(masked_dir + run) for run in brainstem_data]
        corr_labels = ["Music Brainstem", "No Music Brainstem"]

if data_choice == "dmn" :
        neural_runs = [np.load(masked_dir + run) for run in dmn_data]
        corr_labels = ["Music DMNa", "No Music DMNa"]

if data_choice == "random" : 
        neural_runs = []
        for i in np.arange(8) :
                rand_data = np.random.default_rng().uniform(-2, 2, (1500, 3240, 12))
                neural_runs.append(rand_data)
        corr_labels = ["random 1", "random 2", "random 3", "random 4"]

if data_choice == "occipital_pole" :
        neural_runs = [np.load(masked_dir + run) for run in occipital_data]
        corr_labels = ["Music Occipital Pole", "No Music Occipital"]

# unconditionally load in occipital pole data for regression
occ_runs = [np.load(masked_dir + run) for run in occipital_data]

# Prepare the neural data for correlation, grouping together runs, regressing out 
# occipital signal from A1
neural_prepped = []
for i in  np.arange(int(len(neural_runs)/2)) :
       neural_prepped.append(corr_prep(neural_runs[2*i], neural_runs[2*i+1], occ_runs[2*i], occ_runs[2*i+1], regress=True))

# Figure out which rows in the features have zero variance (results in nans in correlation)
n_rows = features[0].shape[0]

# 2D list giving the zero variance rows in each feature
rows = [[i for i in range(n_rows) if np.var(feature[i, :]) == 0.0] for feature in features]

# For comparison, set the dictionary's values as the number zero variance rows in each feature
n_zero_var_rows = [len(rows[i]) for i in np.arange(len(rows))]
zero_var_rows = dict(zip(feat_labels, n_zero_var_rows))

# Chop off the appropriate values in the features and neural data for consistency
for i in np.arange(len(neural_prepped)) : 
        neural_prepped[i] = neural_prepped[i][1:, :]            

for i in np.arange(len(features)) :
        features[i] = features[i][1:, :] 

# Ensure all data has the same number of timepoints
# for main title, we're only interested in looking at a chunk of the 
# neural data
for neural_dat in neural_prepped :
    neural_dat = neural_dat[1095:1176, :]
    print(neural_dat.shape)

time_data = features + neural_prepped
timepoints = time_data[0].shape[0]
for dataset in time_data :
        if dataset.shape[0] != timepoints :
                raise ValueError("All audio features and neural datasets must have the same number of timepoints")


# Perform the RSA

# Choose sliding windows or no sliding windows 
sliding_window = True

if not(isinstance(sliding_window, bool)) : 
        raise TypeError("sliding_window must be a boolean")

# Data parameters
n_feats = len(features)
n_neurdata = len(neural_prepped)
n_RSMs = n_neurdata + 1 # one RSM for each set of neural data + one RSM for the audio
n_trs = neural_prepped[0].shape[0]

# Data storage
RSMs = np.zeros((n_feats, n_RSMs, n_trs, n_trs))
corrs = np.zeros((n_feats, n_neurdata))
if sliding_window :
        sliding_corrs = np.zeros((n_feats, n_neurdata, n_trs-30))
        sliding_stds = np.zeros_like(sliding_corrs)

# Loop over each audio feature
for i, feature in enumerate(features) :
        
        # Loop over each neural dataset
        for j, neurdata in enumerate(neural_prepped) :
                
                if sliding_window :
                        results = RSA(neurdata, feature, sliding_window=True, window_width=30) 
                else :
                        results = RSA(neurdata, feature)

                # Record the current neural RSM
                RSMs[i, j] = results[0]

                # Record the correlations between the current two RSMs
                corrs[i, j] = results[2]

                # Record the sliding correlations betweeen the two current RSMs
                if sliding_window :
                        sliding_corrs[i, j] = results[4]
                        sliding_stds[i, j] = results[5]

        # Record the audio feature RSM in the last index (of second dimension)
        RSMs[i, n_RSMs - 1] = results[1]


# Print the results

print("\nResults")
print("-------")

for i, feature in enumerate(features) :
        
        print("\n")
        print(f"{feat_labels[i]}")
        
        for j in np.arange(n_neurdata) :
                print(f"{corr_labels[j]}: {np.around(corrs[i, j], decimals = 4)}")
        

# Save the average correlations and sliding correlations

save_dir = "/tigress/pnaphade/Eternal_Sunshine/results/RSA/"
roi = "hrf_rA1_regressed"
corrs_path = Path(save_dir + roi + "slide_corrs.npy")
full_slide_corrs_path = Path(save_dir + roi + "full_length_slide_corrs.npy")
        
#if not(corrs_path.exists()) :
#       np.save(corrs_path, corrs)
#if not(full_slide_corrs_path.exists()) :
#       np.save(full_slide_corrs_path, sliding_corrs)




###### Music Scene correlations ######
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
onset_sec, offset_sec = onset_sec-1, offset_sec-1
n_music = len(onset_sec)

# Adjust offset_sec for time
if sliding_window :
         # set last offset equal to the last tr we performed a sliding correlation
        offset_sec[-1] = sliding_corrs.shape[2] - 1
else :
        # set last offset equal to the number of trs after cutting off zero variance
        offset_sec[-1] =  n_trs - 1


# Calculate correlations for music scenes

# If sliding windows were used, we can just pull out the relevant local correlations
if sliding_window :
        
        music_scene_avg_corrs = np.zeros((n_music, n_feats, n_neurdata))
        
        # Pull out the correlations from each music scene
        for onset, offset, i in zip(onset_sec, offset_sec, np.arange(n_music)) :
                scene_corrs = sliding_corrs[:, :, onset:offset]
                scene_avg = np.mean(scene_corrs, axis=2)
                music_scene_avg_corrs[i] = scene_avg

        # Average across all scenes
        music_corrs = np.mean(music_scene_avg_corrs, axis=0)


# Otherwise, we perform traditional RSA on each music scene
else : 
        music_corrs_byscene = np.zeros((n_music, n_feats, n_neurdata))

        # Calculate the correlations for each music scene
        for i in np.arange(n_music) :
                
                # Loop over each audio feature
                for j, feature in enumerate(features) :
        
                        # Loop over each neural dataset
                        for k, neurdata in enumerate(neural_prepped) :
                
                                results = RSA(neurdata[onset_sec[i]:offset_sec[i], :], feature[onset_sec[i]:offset_sec[i], :])

                                music_corrs_byscene[i, j, k] = results[2]

        # Average across all scenes
        music_corrs = np.mean(music_corrs_byscene, axis=0)

# Print music scene results
for h, corrs in enumerate(sliding_corrs_avg) :

        if h == 0 :
                print("Whole-Movie Correlations")
        else :
                print("Music Scene Correlations")

        for i, feature in enumerate(features) :
                
                print(f"{feat_labels[i]}")
                for j in np.arange(n_neurdata) :
                        print(f"{corr_labels[j]}: {np.around(corrs[i, j], decimals=4)}")

                if not(i == 3 and h == 1) :
                        print("\n", end='')
'''

'''
# Plotting
# Visualize the credits regressor
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()
for i, ax in enumerate(axes) :
        
        if i < 4 :
                # plot the music A1 data
                ax.plot(credits_sliding_corrs[i, 0], 'r-', linewidth=1, label=corr_labels[0])
                # plot the no music A1 data
                ax.plot(credits_sliding_corrs[i, 2], 'b-', linewidth=1, label=corr_labels[2])
                ax.set_title(feat_labels[i])
        else :  
                # plot the music A1 data
                ax.plot(credits_sliding_corrs[i-4, 1], 'r-', linewidth=1, label=corr_labels[1])
                # plot the no music A1 data
                ax.plot(credits_sliding_corrs[i-4, 3], 'b-', linewidth=1, label=corr_labels[3]) 
                ax.set_title(feat_labels[i-4])
                ax.set_xlabel("TR")
        
        ax.set_ylim(-0.2, 0.7)  
        
        ax.legend(loc = 'upper left')
        
        if i == 0 or i == 4 :
                ax.set_ylabel("Neural-Audio Correlation")
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
        ax.plot(np.mean(sliding_corrs, axis=0)[i], linewidth=0.5)
        ax.set_title(corr_labels[i])
        ax.set_ylim(-0.2, 1)
        ax.text(5050, 0.8, r'$r_{mean}$' + f" = {np.around(np.mean(corrs, axis=0), decimals=4)[i]}")
                
        if i == 2 or i == 3 :
                ax.set_xlabel("TR")
        
        if i == 0 or i == 2 :
                ax.set_ylabel("Neural-Audio Correlation")
'''

