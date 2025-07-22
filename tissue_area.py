import SimpleITK as sitk
import numpy as np


raw_data = sitk.ReadImage("outputs/l3_slices.nii.gz")
predict_data = sitk.ReadImage("outputs/l3_slices_preds.nii.gz")

spacing = raw_data.GetSpacing()
pixel_area = np.prod(spacing[:-1]) / 100.0 # in cm^2
print(pixel_area)

label_filter = sitk.LabelShapeStatisticsImageFilter()

muscle_classes = {
    'sat': 1,
    'vat': 2,
    'imat': 3,
    'muscle': 4
}

for slc in range(0, predict_data.GetSize()[-1]):
    
    label_filter.Execute(predict_data[..., slc])

    label_areas = {tissue: label_filter.GetPhysicalSize(label) * pixel_area for tissue, label in muscle_classes.items() if label in label_filter.GetLabels()}
    print(f"slc {slc} - {label_areas}")
