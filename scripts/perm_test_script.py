import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load in the data
audio_corr = np.load("audio_corr_chop.npy")
neural_mat = np.load("neural_mat.npy")
real_rvals = np.load("RSM_avg_corrs.npy")


# Function for chopping off desired values from a symmetric matrix
def chop(mat, idx_lo, idx_hi) :
	return mat[idx_lo : idx_hi, idx_lo : idx_hi]

# Variables for sliding window correlations
n_rois = 4
n_trs = audio_corr.shape[0]
window_width = 30
n_windows = n_trs - window_width
n_perms = 1000

# Random generator for shuffling audio data
rng = np.random.default_rng()

# Check for errors in parameters
if n_perms <= 0 :
	raise ValueError("Number of permutations must be greater than 0")

if window_width > n_trs :
	raise ValueError("Window width cannot exceed the length of the dataset")

null_corrs = np.zeros((n_rois, n_perms, n_windows))


# Main loop
for n in range(n_perms) :
	
	# Shuffle the columns of the audio RSM
	rng.shuffle(audio_corr, axis=1)
	
	# Progress	
	print(f"Computing correlations for permutation {n+1} of {n_perms}")	
	
	for i in range(n_windows) :
	
		# Isolate the current window for the neural and audio RSMs
		audio_win_shuf = chop(audio_corr, i, i+window_width)
		neural_win_mat = np.zeros((n_rois, window_width, window_width))
		for j in range(n_rois) :
			neural_win_mat[j] = chop(neural_mat[j], i, i+window_width)
		
			
		# Pull out the upper triangles of the RSMs for correlation
		triu_idx = np.triu_indices(window_width)
		audio_win_triu_shuf = audio_win_shuf[triu_idx]
		neural_win_triu_mat = np.zeros((n_rois, audio_win_triu_shuf.shape[0]))
		for j in range(n_rois) :
			neural_win_triu_mat[j] = neural_win_mat[j][triu_idx]

	
		# Compute the correlations between each ROI and the shuffled audio
		for j in range(n_rois) :
			null_corrs[j, n, i] = stats.pearsonr(neural_win_triu_mat[j], audio_win_triu_shuf)[0]


# Average across windows to produce null distribution
null_corrs_win_avg = np.mean(null_corrs, axis=2)

# Calculate r_diff
perm_means = np.mean(null_corrs_win_avg, axis=1)
r_diff = real_rvals - perm_means
print(f"r_perm values: {perm_means}")
print(f"r_diff values: {r_diff}")


# Calculate z scores for real r values
z_scores = np.zeros(4)
for i, r, r_perm in zip(np.arange(4), real_rvals, perm_means) :
	z_scores[i] = (r-r_perm)/np.std(null_corrs_win_avg, axis=1)[i]	


# Plotting
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()
roi_labels = ["Music Bilateral A1", "Music Right A1", "No Music Bilateral A1", "No Music Right A1"]

for i, ax in zip(np.arange(n_rois), axes) :

	ax.hist(null_corrs_win_avg[i, :], bins=25)
	ax.set_title(roi_labels[i])
	
	# Plot the real r values in red for comparison
	ax.axvline(real_rvals[i], color="r")

	# Add the z scores for the real r values
	ax.text(0.7, 0.8, f"z = {np.around(z_scores[i], decimals=2)}", transform=ax.transAxes)
	
	if i  == 2 or i == 3 :
		ax.set_xlabel("Correlation")

	if i == 0 or i == 2 :
		ax.set_ylabel("Count")
	
#plt.show()

# Function for chopping off desired values from a symmetric matrix
def chop(mat, idx_lo, idx_hi) :
	return mat[idx_lo : idx_hi, idx_lo : idx_hi]
