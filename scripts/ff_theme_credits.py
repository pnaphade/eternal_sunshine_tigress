import numpy as np
import matplotlib.pyplot as plt

# Load in the data
data_dir = "/tigress/pnaphade/Eternal_Sunshine/results/RSA/"
A1_corrs = np.load(data_dir + "hrf_A1_full_length_slide_corrs.npy")
bs_corrs = np.load(data_dir + "hrf_bs_full_length_slide_corrs.npy")

# Pull out the chromagram row
A1_corrs = A1_corrs[2, :, :]
bs_corrs = bs_corrs[2, :, :]

# Music/no music correlations over focus feature theme (rA1, chroma)
fig1, ax = plt.subplots()
ax.plot(A1_corrs[1, 0:13], 'b', label="Music") # music rA1
ax.plot(A1_corrs[3, 0:13], 'r', label="No Music") # no music rA1
ax.set_ylabel("Correlation")
ax.set_xlabel("TR")
ax.legend()
#fig1.show()

# rA1, brainstem, occipital pole correlations over focus feature theam
fig2, ax = plt.subplots()
ax.plot(A1_corrs[1, 0:13],'b', label="Music rA1")
ax.plot(bs_corrs[0, 0:13],'r', label="Music Brainstem")
ax.plot(bs_corrs[1, 0:13], 'g', label="Music Occcipital Pole")
ax.set_xlabel("TR")
ax.legend()
fig2.show()

# Music/no music correlations over credits
fig3, ax = plt.subplots()
ax.plot(A1_corrs[1, -204:], 'b', label="Music") # music rA1
ax.plot(A1_corrs[3, -204:], 'r', label="No Music") # no music rA1
ax.set_ylabel("Correlation")
ax.set_xlabel("TR")
ax.legend()
#fig3.show()
