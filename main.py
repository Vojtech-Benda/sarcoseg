import sys
import argparse
import os
from pathlib import Path
from time import perf_counter
from totalsegmentator.python_api import totalsegmentator
import SimpleITK as sitk
from nnunet.inference import predict
import torch

def main():
    in_dir = Path("./outputs/03.nii.gz")
    seg_out = Path("./outputs/03_temp.nii")

    # os.makedirs(out_dir, exist_ok=True)

    target_vertebrae_map = {
        "vertebrae_L1": 31, 
        "vertebrae_L2": 30, 
        "vertebrae_L3": 29, 
        "vertebrae_L4": 28, 
        "vertebrae_L5": 27, 
        "vertebrae_S1": 26
        }
    print(f"targetting {target_vertebrae_map}")


    
    if not seg_out.exists():
        print("starting total segmentator")
        start = perf_counter()
        totalsegmentator(input=in_dir, output=seg_out, fast=False, ml=True, quiet=False, task="total", roi_subset=list(target_vertebrae_map.keys()), device="gpu")
        duration = perf_counter() - start
        print(f"segmentator finished in {duration}s")
    else:
        print("segmentation image exists")

    seg_image: sitk.Image = sitk.ReadImage(seg_out) == target_vertebrae_map['vertebrae_L3']
    # seg_image = sitk.Flip(seg_image, (False, False, True))
    dicom_image: sitk.Image = sitk.ReadImage(in_dir)
    
    label_filter = sitk.LabelShapeStatisticsImageFilter()
    label_filter.Execute(seg_image)
    print(label_filter.GetLabels())
    
    l3_centroid = label_filter.GetCentroid(1)
    l3_centroid_idx = seg_image.TransformPhysicalPointToIndex(l3_centroid)
    print(f"phys {l3_centroid} idx {l3_centroid_idx}")


    seg_image_sagittal = seg_image[l3_centroid_idx[0], ...]
    l3_components_image = sitk.ConnectedComponent(seg_image_sagittal)
    label_filter.Execute(l3_components_image)
    
    l3_body_label = max(label_filter.GetLabels(), key=lambda l: label_filter.GetNumberOfPixels(l))

    l3_body: sitk.Image = l3_components_image == l3_body_label
    
    label_filter.Execute(l3_body)
    l3_body_centroid = label_filter.GetCentroid(1)
    l3_body_centroid_idx = l3_body.TransformPhysicalPointToIndex(l3_body_centroid)

    l3_slice_idxs = (l3_body_centroid_idx[1] - 15, l3_body_centroid_idx[1] + 15)
    l3_slices = dicom_image[..., l3_slice_idxs[0]:l3_slice_idxs[1]]
    l3_slices_path = Path("outputs", "l3_slices_win.nii.gz")
    sitk.WriteImage(l3_slices, l3_slices_path)
    
    target_tissues = {
        "muscle": 1, 
        "sat": 2, 
        "vat": 3, 
        "imat": 4
        }
    
    l3_preds_path = Path("outputs", "l3_slices_preds_win.nii.gz")
    model_dir = Path("models", "muscle_fat_tissue_stanford_0_0_2")
    f = predict.predict_cases(model=str(model_dir), 
                          list_of_lists=[[l3_slices_path]], 
                          output_filenames=[l3_preds_path],
                          folds="all",
                          save_npz=False,
                          num_threads_preprocessing=8,
                          num_threads_nifti_save=8,
                          segs_from_prev_stage=None,
                          do_tta=False,
                          mixed_precision=True,
                          overwrite_existing=True,
                          all_in_gpu=False,
                          step_size=0.5,
                          checkpoint_name="model_final_checkpoint",
                          segmentation_export_kwargs=None,
                          disable_postprocessing=True
                          )
    
    
if __name__ == "__main__":
    main()
    