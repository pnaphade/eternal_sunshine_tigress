import numpy as np
import matplotlib.pyplot as plt

# Load in the data (chromagram is the 3rd feature)
data_dir = "/tigress/pnaphade/Eternal_Sunshine/results/RSA/"
# which audio feature we want to use
f = 0
A1_corrs = np.load(data_dir + "hrf_A1_full_length_slide_corrs.npy")[f, :, :]
rA1_stds = np.load(data_dir + "rA1_slide_stds.npy")[f, :, :]
rA1_regcorrs = np.load(data_dir + "rA1_reg_slide_corrs.npy")[f, :, :]
rA1_regstds = np.load(data_dir+ "rA1_reg_slide_stds.npy")[f, :, :]

bs_corrs = np.load(data_dir + "hrf_bs_full_length_slide_corrs.npy")[f, :, :]
bs_stds = np.load(data_dir + "bs_slide_stds.npy")[f, :, :]
bs_regcorrs = np.load(data_dir + "bs_reg_slide_corrs.npy")[f, :, :]
bs_regstds = np.load(data_dir + "bs_reg_slide_stds.npy")[f, :, :]

dmn_corrs = np.load(data_dir + "hrf_dmnA_full_length_slide_corrs.npy")[f, :, :]
dmn_stds = np.load(data_dir + "dmn_slide_stds.npy")[f, :, :]
dmn_regcorrs = np.load(data_dir + "dmn_reg_slide_corrs.npy")[f, :, :]
dmn_regstds = np.load(data_dir + "dmn_reg_slide_stds.npy")[f, :, :]

# Music/no music correlations over focus feature theme and theme
# 0:00 to 1:35, but only go in 60s because of sliding windows and not including
#first 5 s of silence in the movie (rA1, chroma)
t1 = 1051 # how far into the movie should we look?
t2 = 1117
rA1mus_rc = rA1_regcorrs[0, t1:t2]
rA1nomus_rc = rA1_regcorrs[1, t1:t2]
rA1mus_rstd = rA1_regstds[0, t1:t2]/2
rA1nomus_rstd = rA1_regstds[1, t1:t2]/2

rA1mus_c = A1_corrs[0, t1:t2]
rA1nomus_c = A1_corrs[1, t1:t2]
rA1mus_std = rA1_stds[0, t1:t2]
rA1nomus_std = rA1_stds[1, t1:t2]

del_t = t2-t1
time = np.linspace(0, del_t, del_t)
fig1, ax = plt.subplots()
ax.plot(time, rA1mus_rc, 'b', label="Music, regressed") # music rA1
ax.fill_between(time, rA1mus_rc+rA1mus_rstd, rA1mus_rc-rA1mus_rstd, color='b', alpha=0.1) 
ax.plot(time, rA1nomus_rc, 'r', label="No Music, regressed") # no music rA1
ax.fill_between(time, rA1nomus_rc+rA1nomus_rstd,
        rA1nomus_rc-rA1nomus_rstd, color='r', alpha=0.1) 
ax.set_ylabel("Correlation")
ax.set_xlabel("TR")
ax.legend()
#fig1.show()

# rA1, brainstem, dmn, occipital pole correlations over focus feature theme
'''
bsmus_rc = bs_regcorrs[0, 0:13]
bsnomus_rc = bs_regcorrs[1, 0:13]
bsmus_rstd = bs_regstds[0, 0:13]/2
bsnomus_rstd = bs_regstds[1, 0:13]/2

bsmus_c = bs_corrs[0, 0:13]
bsnomus_c = bs_corrs[1, 0:13]
bsmus_std = bs_stds[0, 0:13]
bsnomus_std = bs_stds[1, 0:13]

dmnmus_rc = dmn_regcorrs[0, 0:13]
dmnnomus_rc = dmn_regcorrs[1, 0:13]
dmnmus_rstd = dmn_regstds[0, 0:13]/2
dmnnomus_rstd = dmn_regstds[1, 0:13]/2

dmnmus_c = dmn_corrs[0, 0:13]
dmnnomus_c = dmn_corrs[1, 0:13]
dmnmus_std = dmn_stds[0, 0:13]
dmnnnomus_std = dmn_stds[1, 0:13]

fig2, ax = plt.subplots()
ax.plot(time, rA1mus_rc,'b', label="Music rA1, regressed")
ax.fill_between(time, rA1mus_rc+rA1mus_rstd, rA1mus_rc-rA1mus_rstd, color='b', alpha=0.1) 
ax.plot(time, bsmus_rc,'r', label="Music Brainstem, regressed")
ax.fill_between(time, bsmus_rc+bsmus_rstd, bsmus_rc-bsmus_rstd, color='r', alpha=0.1) 
#ax.plot(, 'g', label="Music Occcipital Pole")
#ax.fill_between(time, rA1mus+rA1mus_std, rA1mus-rA1mus_std, color='b', alpha=0.1) 
ax.plot(time, dmnmus_rc, 'm', label="Music DMNa, regressed")
ax.fill_between(time, dmnmus_rc+dmnmus_rstd, dmnmus_rc-dmnmus_rstd, color='m', alpha=0.1) 
ax.set_xlabel("TR")
ax.set_ylabel("Correlation")
ax.set_ylim(-0.46, 0.12)
ax.legend(loc = 'upper center', bbox_to_anchor=(0.5, 1.15))
#fig2.show()

# Music/no music correlations over credits
fig3, ax = plt.subplots()
ax.plot(A1_corrs[1, -204:], 'b', label="Music rA1")
ax.plot(bs_corrs[0, -204:], 'r', label="Music Brainstem")
ax.plot(bs_corrs[1, -204:], 'g', label="Music Occipital Pole")
ax.plot(dmn_corrs[0, -204:], 'm', label="Music DMNa")
ax.set_ylabel("Correlation")
ax.set_xlabel("TR")
ax.legend()

#fig3.show()
'''
