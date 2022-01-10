import numpy as np
import matplotlib.pyplot as plt

# Load in the sliding correlations
A1_corrs = np.load('/tigress/pnaphade/Eternal_Sunshine/results/RSA/hrf_A1_slide_corrs.npy')
brainstem_corrs = np.load('/tigress/pnaphade/Eternal_Sunshine/results/RSA/hrf_bs_slide_corrs.npy')

# Load in the whole-movie correlations
A1_whole_corrs = np.load('/tigress/pnaphade/Eternal_Sunshine/results/RSA/hrf_A1_full_length_slide_corrs.npy')
bs_whole_corrs = np.load('/tigress/pnaphade/Eternal_Sunshine/results/RSA/hrf_bs_full_length_slide_corrs.npy')



# Plot feature-based results
labels = ["Mel Spect", "Spect", "Chroma", "MFCC"]
mus_corr_vals = [A1_corrs[0, 1], A1_corrs[1, 1], A1_corrs[2, 1], A1_corrs[3,1]]

# Compute standard deviations for error bars
feat_err = np.zeros(4)
for i in np.arange(4) :
	feat_err[i] = np.std(A1_whole_corrs[i, 1, :])


fig1 = plt.figure(figsize=(6, 4))
ax = fig1.add_axes([0.1, 0.1, 0.6, 0.75])
x = np.arange(len(labels))

# Set up location of bars
feat_bars = ax.bar(x, mus_corr_vals, width=0.6, label='rA1 Music', yerr = feat_err, capsize=10)

# Aesthetics
ax.set_ylabel("Correlation")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(bbox_to_anchor=(1.04, 1), loc=2, borderaxespad=0.)

fig1.show()


# Plot ROI-based results
labels = ["Music", "No Music"]
A1_corr_vals = [A1_corrs[0, 0], A1_corrs[0, 2]]
rA1_corr_vals = [A1_corrs[0, 1], A1_corrs[0, 3]]
bs_corr_vals = [brainstem_corrs[0, 0], brainstem_corrs[0, 2]]

# Compute standard deviations for error bars
A1_err = [np.std(A1_whole_corrs[2, 0, :]), np.std(A1_whole_corrs[2, 2, :])]
rA1_err = [np.std(A1_whole_corrs[2, 1, :]), np.std(A1_whole_corrs[2, 3, :])]
bs_err = [np.std(bs_whole_corrs[2, 0, :]), np.std(bs_whole_corrs[2, 2, :])]


fig2 = plt.figure(figsize=(5.5, 4))
ax = fig2.add_axes([0.1, 0.1, 0.6, 0.75])
x = np.arange(len(labels))
width = 0.15

# Set up location of bars
A1_bars = ax.bar(x - width, A1_corr_vals, width, label='A1', yerr=A1_err, capsize=5)
rA1_bars = ax.bar(x, rA1_corr_vals, width, label='rA1', yerr=rA1_err, capsize=5)
bs_bars = ax.bar(x + width, bs_corr_vals, width, label='Brainstem', yerr=bs_err, capsize=5)

# Aesthetics
ax.set_ylabel("Correlation")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

fig2.show()
