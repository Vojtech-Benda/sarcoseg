from typing import Union
from pathlib import Path
import os
import glob
from time import perf_counter
from tqdm import tqdm

from totalsegmentator.python_api import totalsegmentator
import SimpleITK as sitk
import numpy as np
import nibabel as nib
from nnunet.inference.predict import predict_cases

TARGET_VERTEBRAE_MAP = {
    "vertebrae_L1": 31, 
    "vertebrae_L2": 30, 
    "vertebrae_L3": 29, 
    "vertebrae_L4": 28, 
    "vertebrae_L5": 27, 
    "vertebrae_S1": 26
    }


TARGET_TISSUES_MAP = {
    "sat": 1, 
    "vat": 2, 
    "imat": 3,
    "muscle": 4 
}


MODEL_DIR = Path("models", "muscle_fat_tissue_stanford_0_0_2")


def segment_spine(input_dir: Union[str, Path], 
                 output_dir: Union[str, Path] = None, 
                 **kwargs):
    
    if not None:
        os.makedirs(output_dir, exist_ok=True)
    
    data_file_list = glob.glob(f"{input_dir}/*.nii.gz")
    
    runtime_duration = {}
    for data_file in data_file_list:
        data_file = Path(data_file)
        print(f"\nsegmenting vertebrae for {data_file.name}")
        
        start = perf_counter()
        pred_mask: nib.nifti1.Nifti1Image = totalsegmentator(data_file, 
                                    fast=False, 
                                    ml=True,
                                    quiet=True,
                                    task="total",
                                    roi_subset=list(TARGET_VERTEBRAE_MAP.keys()),
                                    device="gpu")
        runtime_duration['spine_seg'] = perf_counter() - start
        print(f"vertebrae segmentation finised in {runtime_duration['spine_seg']:.2f}s")
        
        pred_mask = nib.funcs.as_closest_canonical(pred_mask)

        data_outdir = Path(output_dir, data_file.stem.removesuffix(".nii"))
        os.makedirs(data_outdir, exist_ok=True)
        
        spine_pred_filepath = data_outdir.joinpath("spine_pred_mask.nii.gz")
        try:
            nib.save(pred_mask, spine_pred_filepath)
            print(f"saved spine segmentation mask to '{spine_pred_filepath}'")
        except RuntimeError as err:
            print(err)
        # arr = nibabel_mask.get_fdata().astype(np.uint8)
        
        pred_volume = sitk.ReadImage(spine_pred_filepath)
        pred_volume = sitk.DICOMOrient(pred_volume, "RAS")
        input_volume = sitk.ReadImage(data_file)
        input_volume = sitk.DICOMOrient(input_volume, "RAS")
        
        l3_volume = pred_volume == TARGET_VERTEBRAE_MAP['vertebrae_L3']

        start = perf_counter()
        label_filter = sitk.LabelShapeStatisticsImageFilter()
        label_filter.Execute(l3_volume)

        centroid_l3 = label_filter.GetCentroid(1) # there is only one non-zero label
        centroid_l3_index = l3_volume.TransformPhysicalPointToIndex(centroid_l3)
        
        sagittal_slice = l3_volume[centroid_l3_index[0], ...]

        components_l3 = sitk.ConnectedComponent(sagittal_slice)
        label_filter.Execute(components_l3)
        
        # get the 2 largest components in sagittal slice
        # should correspond to vertebre body and vertebre spine
        largest_components = sorted(
            label_filter.GetLabels(), 
            key=lambda label: label_filter.GetNumberOfPixels(label),
            reverse=True
            )[:2]
        
        component_centroids = {comp: label_filter.GetCentroid(comp) for comp in largest_components}
        
        # get the L3 body centroid which is to the left of whole vertebre centroid, ie in front of it 
        for label, comp_center in component_centroids.items():
            if comp_center[0] <= centroid_l3[1]:
                # get the centroid's index
                l3_body_centroid_index = sagittal_slice.TransformPhysicalPointToIndex(comp_center)        
        
        runtime_duration['vert_body_select'] = perf_counter() - start
        
        slices_num = kwargs.get("slices_num", 0)
        if slices_num > 0:
            z_size = input_volume.GetSize()[-1]
            slices_range = (l3_body_centroid_index[-1] - slices_num, l3_body_centroid_index[-1] + slices_num)
            
            if slices_range[0] < 0:
                print(f"lower index {slices_range[0]} is outside extent for Z dimension, setting to 0")
                slices_range[0] = 0
            elif slices_range[1] > z_size:
                slices_range[1] = z_size  
                print(f"upper index {slices_range[1]} is outside extent for Z dimension {z_size}, setting to {z_size}")
                           
            print(f"extracting {slices_range[1] - slices_range[0]} slices in range {slices_range}, middle slice index is {l3_body_centroid_index[-1]}")
            l3_slices_volume = input_volume[..., slices_range[0]:slices_range[1] + 1]
        else:
            print(f"extracting single slice at {l3_body_centroid_index}")
            l3_slices_volume = input_volume[..., l3_body_centroid_index[-1]]
            
        l3_slices_filepath = data_outdir.joinpath("l3_slices.nii.gz")
        try:
            sitk.WriteImage(l3_slices_volume, l3_slices_filepath)
        except RuntimeError as err:
            print(err)
        print(f"saved L3 volume slices to '{l3_slices_filepath}'")
        
    return {'l3_body_centroid_index': l3_body_centroid_index, 'runtime_duration': runtime_duration}
         
         
def segment_tissues(input_dir: Union[str, Path],
                    output_dir: Union[str, Path],
                    metrics: list = None, **kwargs):
    
    if not isinstance(input_dir, Path):
        input_dir = Path(input_dir)
    
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)
    
    l3_slices_filepath = input_dir.joinpath("l3_slices.nii.gz")
    l3_muscle_pred_filepath = output_dir.joinpath("l3_tissue_preds.nii.gz")
    print(l3_muscle_pred_filepath)
    print(f"\nstarting tissue segmentation for {input_dir}")
    runtime_duration = {}
    
    start = perf_counter()
    predict_cases(
        model=str(MODEL_DIR),
        list_of_lists=[[l3_slices_filepath]],
        output_filenames=[l3_muscle_pred_filepath],
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
    
    runtime_duration['tissue_seg'] = perf_counter() - start
    print(f"tissue segmentation finished in {runtime_duration['tissue_seg']}")
    
    # tissue_masks = sitk.ReadImage(l3_muscle_pred_filepath)
    # label_filter = sitk.LabelShapeStatisticsImageFilter()
    # label_filter.Execute(tissue_masks)
    # print(f"tissue labels: {label_filter.GetLabels()}")
    