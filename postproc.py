import SimpleITK as sitk

image = sitk.ReadImage("outputs/l3_slices_preds.nii.gz")

binary_fill = sitk.Image(image.GetSize(), image.GetPixelIDValue(), image.GetNumberOfComponentsPerPixel())
gray_fill = sitk.Image(image.GetSize(), image.GetPixelIDValue(), image.GetNumberOfComponentsPerPixel()) 

labels = (1, 2, 3, 4)

for slc in range(image.GetDepth()):
    
    gray_fill[..., slc] = sitk.GrayscaleFillhole(image[..., slc])
    
    for l in labels:
        lab_im = image[..., slc] == l
        binary_fill[..., slc] = sitk.BinaryFillhole(lab_im) * l
        
    

median_filt = sitk.Median(image, radius=(5, 5, 5))
    

gray_fill = sitk.Resample(gray_fill, referenceImage=image)
binary_fill = sitk.Resample(binary_fill, referenceImage=image)

sitk.WriteImage(median_filt, "outputs/median_filt.nii.gz")
sitk.WriteImage(gray_fill, "outputs/gray_fill.nii.gz")
sitk.WriteImage(binary_fill, "outputs/binary_fill.nii.gz")

