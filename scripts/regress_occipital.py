import numpy as np
from sklearn import linear_model


# Load in occipital data and rA1 data -- first run only
datadir = "/tigress/pnaphade/Eternal_Sunshine/scripts/rois/masked_data/"
rA1_music = np.load(datadir + "music/rA1_run1_n12.npy")
rA1_nomusic = np.load(datadir + "no_music/rA1_run1_n11.npy")
occ_music = np.load(datadir + "music/occipital_pole_run1_n12.npy")
occ_nomusic = np.load(datadir + "no_music/occipital_pole_run1_n11.npy")


# Regress out occipital pole signal for music subjects
for i in np.arange(rA1_music.shape[2]) :

	# Pull out the data for each subject
	occ_music_sub = occ_music[:, :, i]
	rA1_music_sub = rA1_music[:, :, i]

	# Progress update
	print("Performing regression on music subject " + str(i))

	# Create and fit the regression model using occipital pole as the predictor
	reg = linear_model.LinearRegression()
	reg.fit(occ_music_sub.T, rA1_music_sub.T) # transpose so we have time x voxels

	# Subtract the part of the rA1 signal that can be explained by occipital pole signal
	rA1_music_reg = rA1_music_sub - np.dot(reg.coef_, occ_music_sub) - reg.intercept_[:, np.newaxis]


	# Save the results
	rA1_music_regressed = np.zeros_like(rA1_music)
	rA1_music_regressed[:, :, i] = rA1_music_reg

# Regress out occipital pole signal for no music subjects
