import SimpleITK as sitk

image = sitk.ReadImage("outputs/l3_slices_preds.nii.gz")

post_image = sitk.Image(image.GetSize(), image.GetPixelIDValue(), image.GetNumberOfComponentsPerPixel())
for slc in range(image.GetDepth()):
    
    post_image[..., slc] = sitk.GrayscaleFillhole(image[..., slc])
    
sitk.LabelVoting()
post_image = sitk.Resample(post_image, referenceImage=image)
print(post_image.GetOrigin(), post_image.GetSpacing(), post_image.GetDirection())
sitk.WriteImage(post_image, "outputs/binaryfill.nii.gz")