import numpy as np
import nibabel as nib

maskdir = '/tigress/pnaphade/Eternal_Sunshine/scripts/rois/'

# Choose desired  mask
roi = maskdir + 'rA1_mask_25mm.nii.gz'
roiname = 'rA1'

# music group
music_subjs = ['sub-002','sub-003','sub-005','sub-008','sub-010','sub-011','sub-013','sub-015','sub-017','sub-019','sub-021','sub-023', 'sub-025', 'sub-027','sub-030', 'sub-032', 'sub-034', 'sub-035', 'sub-037', 'sub-039', 'sub-042', 'sub-044', 'sub-046', 'sub-048', 'sub-050']

no_music_subjs = ['sub-001','sub-004','sub-006','sub-007','sub-009','sub-012','sub-014','sub-016','sub-018','sub-020','sub-022', 'sub-024', 'sub-026', 'sub-028', 'sub-029', 'sub-031', 'sub-033', 'sub-036', 'sub-038', 'sub-040', 'sub-041', 'sub-043', 'sub-045', 'sub-047', 'sub-049']

watch_fn = ['_clean_watch_run1_smooth.nii.gz','_clean_watch_run2_smooth.nii.gz']


datadir = '/tigress/jamalw/Eternal_Sunshine/data/'

mask = nib.load(roi).get_fdata()
mask_size = np.nonzero(mask)[0].shape

# mask data from music group
music_masked_run1 = np.zeros((mask_size[0],3240,len(music_subjs)))
music_masked_run2 = np.zeros((mask_size[0],3212,len(music_subjs)))

for i in range(0,len(music_subjs)):
    run1 = nib.load(datadir + music_subjs[i] + '/' + music_subjs[i] + watch_fn[0]).get_fdata()[...,0:3240]
    run2 = nib.load(datadir + music_subjs[i] + '/' + music_subjs[i] + watch_fn[1]).get_fdata()[...,3:]
    print(music_subjs[i], ' Data Loaded')   
 
    music_masked_run1[...,i] = run1[mask == 1] 
    music_masked_run2[...,i] = run2[mask == 1]
    print(music_subjs[i], ' Data Masked')

np.save(maskdir + 'masked_data/music/' + roiname + '_run1_n' + str(len(music_subjs)),music_masked_run1)

np.save(maskdir + 'masked_data/music/' + roiname + '_run2_n' + str(len(music_subjs)),music_masked_run2)

# mask data from no-music group
no_music_masked_run1 = np.zeros((mask_size[0],3240,len(no_music_subjs)))
no_music_masked_run2 = np.zeros((mask_size[0],3212,len(no_music_subjs)))

for i in range(0,len(no_music_subjs)):
    run1 = nib.load(datadir + no_music_subjs[i] + '/' + no_music_subjs[i] + watch_fn[0]).get_fdata()[...,0:3240]
    run2 = nib.load(datadir + no_music_subjs[i] + '/' + no_music_subjs[i] + watch_fn[1]).get_fdata()[...,3:]
    print(no_music_subjs[i], ' Data Loaded')   
 
    no_music_masked_run1[...,i] = run1[mask == 1] 
    no_music_masked_run2[...,i] = run2[mask == 1]
    print(no_music_subjs[i], ' Data Masked')

np.save(maskdir + 'masked_data/no_music/' + roiname + '_run1_n' + str(len(no_music_subjs)),no_music_masked_run1)

np.save(maskdir + 'masked_data/no_music/' + roiname + '_run2_n' + str(len(no_music_subjs)),no_music_masked_run2)


   
