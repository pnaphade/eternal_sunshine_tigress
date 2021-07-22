import numpy as np
import nibabel as nib
from nilearn.image import resample_img, math_img
import nilearn

input_fn = 'brain_stem_mask'

# load output affine matrix
output_affine = nib.load('/tigress/jamalw/Eternal_Sunshine/scripts/rois/MNI152_T1_25mm_brain.nii.gz').affine

# load data to be resampled
input_data = nib.load('/tigress/jamalw/Eternal_Sunshine/scripts/rois/' + input_fn + '.nii.gz')

img_in_mm_space = resample_img(input_data, target_affine=output_affine,
                               target_shape=(78, 93, 78))

# binarize mask
mask_no_bin = img_in_mm_space.get_data().flatten()
mask_no_bin[mask_no_bin < 1] = 0
mask_bin = np.reshape(mask_no_bin, (78, 93, 78))

# convert to nifti image
maxval = np.max(mask_bin)
minval = np.min(mask_bin)
img = nib.Nifti1Image(mask_bin, affine=output_affine)
img.header['cal_min'] = minval
img.header['cal_max'] = maxval

# smooth and re-binarize mask
img_smooth = nilearn.image.smooth_img(img,fwhm='fast')
img_smooth_bin = math_img('img > 0', img=img_smooth)

# save mask
nib.save(img_smooth_bin, '/tigress/jamalw/Eternal_Sunshine/scripts/rois/' + input_fn + '_25mm_smooth_bin.nii.gz')


