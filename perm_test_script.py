import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load in the data
audio_corr = np.load("audio_corr_chop.npy")
neural_triu_mat = np.load("neural_triu_mat.npy")
real_rvals = np.load("RSM_corrs.npy")[0, :]

# Number of permutations
n_perms = 1000

# Number of rois
n_rois = 4

# Matrix for storing null distributions
null_corrs = np.zeros((n_rois, n_perms))

# Random generator for shuffling audio data
rng = np.random.default_rng()

for i in range(n_perms) :
	
	# Shuffle the columns of the audio RSM
	rng.shuffle(audio_corr, axis=1)
	
	# Pull out the upper triangle for correlation
	audio_triu_shuf = audio_corr[np.triu_indices(audio_corr.shape[0])]

	# Compute the correlations between each ROI and the shuffled audio
	for j in range(n_rois) :
		
		print("Computing correlation %d of %d" % (j+1+(i*n_rois), n_perms*n_rois))
		null_corrs[j, i] = stats.pearsonr(neural_triu_mat[j], audio_triu_shuf)[0]



# Plotting
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()
fig.suptitle("Null Distributions for Neural-Audio RSM Correlation", fontweight="bold")
roi_labels = ["Music Bilateral A1", "Music Right A1", "No Music Bilateral A1", "No Music Right A1"]

for i, ax in zip(np.arange(n_rois), axes) :

	ax.hist(null_corrs[i, :], bins = 25)
	ax.set_title(roi_labels[i])
	
	# Plot the real r values in red for comparison
	ax.axvline(real_rvals[i], color="r")
	
	if i  == 2 or i == 3 :
		ax.set_xlabel("Correlation")

	if i == 0 or i == 2 :
		ax.set_ylabel("Count")
	
#plt.show()


# Calculating r_diff

r_diff = real_rvals - np.mean(null_corrs, axis=1)
print(f"r_diff values: {r_diff}")
