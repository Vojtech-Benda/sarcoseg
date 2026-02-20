# %%
import skimage as sk
import SimpleITK as sitk
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline
from src.utils import compute_metrics, read_volume, DEFAULT_TISSUE_CLASSES

# %%
def imshow(arr):
    plt.imshow(arr)
    plt.show()

# %%
mask_path = "tests\\output\\1.3.6.1.4.1.36302.1.1.2.67386.4681372\\1.3.12.2.1107.5.1.4.75968.30000025063005325811100017182\\tissue_mask.nii.gz"
volume_path = "tests\\output\\1.3.6.1.4.1.36302.1.1.2.67386.4681372\\1.3.12.2.1107.5.1.4.75968.30000025063005325811100017182\\tissue_slices.nii.gz"
mask_nifti = read_volume(mask_path)
volume_nifti = read_volume(volume_path)
mask_arr = mask_nifti.image.get_fdata()
volume_arr = volume_nifti.image.get_fdata()
spacing = volume_nifti.spacing
pixel_size = np.prod(spacing[:2]) / 100.0 # in cm^2,
print(f"{pixel_size=}")
imshow(mask_arr)
# %%
_, counts = np.unique_counts(mask_arr)
area = {label: cnt * pixel_size for label, cnt in zip(DEFAULT_TISSUE_CLASSES.keys(), counts[1:])} # omit 0 - background
area
# %%
sitk_mask = sitk.ReadImage(mask_path)
sitk_volume = sitk.ReadImage(volume_path)

# %%
stats_filter = sitk.LabelShapeStatisticsImageFilter()
stats_filter.Execute(sitk_mask[..., 0])
np.array([stats_filter.GetPhysicalSize(l) for l in stats_filter.GetLabels()]) / 100.0
orient = sitk.DICOMOrient(sitk_mask, "LPS")
imshow(sitk.GetArrayViewFromImage(orient)[0, ...])
# %%
# skimage region properties
np.array([region['area'] for region in sk.measure.regionprops(mask_arr.astype(np.uint8)[..., 0], spacing=spacing[:-1])]) / 100.
# %%
# displaying image and label overlay
mask_lps = sitk.DICOMOrient(sitk_mask, "LPS")
volume_lps = sitk.DICOMOrient(sitk_volume, "LPS")

red = [255, 0, 0]
yellow = [255, 255, 0]
green = [0, 255, 0]
purple = [128, 0, 128]
width, level = 400, 50
tissue_window = (level - width // 2, level + width // 2)

volume_windowed = sitk.Cast(sitk.IntensityWindowing(volume_lps, tissue_window[0], tissue_window[1]), sitk.sitkUInt8)
volume_uint8 = sitk.Cast(sitk.RescaleIntensity(volume_lps), sitk.sitkUInt8)
sitk_overlay = sitk.LabelOverlay(volume_uint8, mask_lps, opacity=0.5, colormap=purple + red + yellow + green)
imshow(sitk.GetArrayViewFromImage(sitk_overlay)[0, ...])
