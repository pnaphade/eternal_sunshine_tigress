import numpy as np
from sklearn import linear_model


# Load in occipital data and rA1 data -- first run only
datadir = "/tigress/pnaphade/Eternal_Sunshine/scripts/rois/masked_data/"
rA1_music_run1 = np.load(datadir + "music/rA1_run1_n12.npy")
rA1_nomusic_run1 = np.load(datadir + "no_music/rA1_run1_n11.npy")
occ_music_run1 = np.load(datadir + "music/occipital_pole_run1_n12.npy")
occ_nomusic_run1 = np.load(datadir + "no_music/occipital_pole_run1_n11.npy")
rA1_music_run2 = np.load(datadir + "music/rA1_run2_n12.npy")
rA1_nomusic_run2 = np.load(datadir + "no_music/rA1_run2_n11.npy")
occ_music_run2 = np.load(datadir + "music/occipital_pole_run2_n12.npy")
occ_nomusic_run2 = np.load(datadir + "no_music/occipital_pole_run2_n11.npy")

# Arrays for saving cleaned rA1 signal
rA1_music_reg_run1 = np.zeros_like(rA1_music_run1)
rA1_music_reg_run2 = np.zeros_like(rA1_music_run2)
rA1_nomusic_reg_run1 = np.zeros_like(rA1_nomusic_run1)
rA1_nomusic_reg_run2 = np.zeros_like(rA1_nomusic_run2)

# Regress out occipital pole signal from music group rA1
for occ_run, rA1_run, rA1_regressed, i in zip([occ_music_run1, occ_music_run2], [rA1_music_run1, rA1_music_run2], [rA1_music_reg_run1, rA1_music_reg_run2], np.arange(2)) : 

	# Regress out occipital pole signal for music subjects
	for subj in np.arange(rA1_run.shape[2]) :
		
		# Pull out the data for each subject
		occ_music_sub = occ_run[:, :, subj]
		rA1_music_sub = rA1_run[:, :, subj]

		# Progress update
		print("Performing regression on music subject " + str(subj) + ", run " + str(i))

		# Create and fit the regression model using occipital pole as the predictor
		reg = linear_model.LinearRegression()
		reg.fit(occ_music_sub.T, rA1_music_sub.T) # transpose so we have time x voxels

		# Subtract the part of the rA1 signal that can be explained by occipital pole signal
		rA1_music_reg = rA1_music_sub - np.matmul(reg.coef_, occ_music_sub) - reg.intercept_[:, np.newaxis]

		# Record the results
		rA1_regressed[:, :, subj] = rA1_music_reg


# Regress out occipital pole signal from no music group rA1
for occ_run, rA1_run, rA1_regressed, i in zip([occ_nomusic_run1, occ_nomusic_run2], [rA1_nomusic_run1, rA1_nomusic_run2], [rA1_nomusic_reg_run1, rA1_nomusic_reg_run2], np.arange(2)) : 

	# Regress out occipital pole signal for music subjects
	for subj in np.arange(rA1_run.shape[2]) :
		
		# Pull out the data for each subject
		occ_nomusic_sub = occ_run[:, :, subj]
		rA1_nomusic_sub = rA1_run[:, :, subj]

		# Progress update
		print("Performing regression on no music subject " + str(subj) + ", run " + str(i))

		# Create and fit the regression model using occipital pole as the predictor
		reg = linear_model.LinearRegression()
		reg.fit(occ_nomusic_sub.T, rA1_nomusic_sub.T) # transpose so we have time x voxels

		# Subtract the part of the rA1 signal that can be explained by occipital pole signal
		rA1_nomusic_reg = rA1_nomusic_sub - np.matmul(reg.coef_, occ_nomusic_sub) - reg.intercept_[:, np.newaxis]

		# Record the results
		rA1_regressed[:, :, subj] = rA1_nomusic_reg


# Save the rA1 signal with occipital pole regressed out
np.save(datadir + "music/rA1_regressed_run1.npy", rA1_music_reg_run1)
np.save(datadir + "no_music/rA1_regressed_run1.npy", rA1_nomusic_reg_run1)
np.save(datadir + "music/rA1_regressed_run2.npy", rA1_music_reg_run2)
np.save(datadir + "no_music/rA1_regressed_run2.npy", rA1_nomusic_reg_run2)

