from typing import Union
from pathlib import Path
from time import perf_counter

from totalsegmentator.python_api import totalsegmentator
import numpy as np
from numpy import typing as npt

import nibabel as nib
import skimage as sk
from nnunet.inference.predict import predict_cases

from src import segmentation

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

TISSUE_LABEL_INDEX = list(TARGET_TISSUES_MAP.keys())

MODEL_DIR = Path("models", "muscle_fat_tissue_stanford_0_0_2")


def segment_ct(
    input_dir: str,
    output_dir: str,
    additional_metrics: list,
    **kwargs
    ):
    case_dirs: list[Path] = list(Path(input_dir).glob("*/"))
    print(f"found {len(case_dirs)} cases")

    for case_dir in case_dirs:
        case_output_dir = Path(output_dir, case_dir.name)
        case_output_dir.mkdir(exist_ok=True)

        ct_volume_paths = list(case_dir.glob("*.nii.gz"))
        print(f"found {len(ct_volume_paths)} volumes to segment spine for case {case_dir.name}")
        
        slices_num = kwargs.get('slices_num', 0)
        save_segmentations = kwargs.get('save_segmentations')
        
        for ct_volume_path in ct_volume_paths:
            spine_results = segmentation.segment_spine(
                ct_volume_path,
                case_output_dir,
                )

            spine_mask = spine_results['spine_mask'] if 'spine_mask' in spine_results else spine_results['spine_mask_path']

            slice_results = segmentation.extract_slices(
                ct_volume_path,
                case_output_dir,
                spine_mask,
                slices_num
                )

            tissue_results = segmentation.segment_tissues(
                slice_results['sliced_volume_path'],
                case_output_dir
                )

            postproc_results = segmentation.postprocess_tissue_masks(
                tissue_results['mask'],
                tissue_results['volume'],
                # tissue_results['mask_affine'],
                tissue_results['mask_filepath']
            )

            metric_results = segmentation.compute_metrics(
                    postproc_results['processed_masks'],
                    metrics=additional_metrics,
                    spacing=tissue_results.get('spacing', None)
                )
            print(metric_results['area'])


def segment_spine(
    input_nifti_path: Union[str, Path], 
    output_dir: Union[str, Path] = None,
    vert_classes: list = None,
    overwrite_output: bool = False
    ) -> dict:
    """
    Segment spine vertebrae.

    Args:
        input_nifti_path (Union[str, Path]): path to nifti file
        output_dir (Union[str, Path], optional): directory to store segmentation mask. Defaults to None.

    Returns:
        spine_results (dict): spine segmentation results
    """    
    
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)
    
    spine_mask_path = output_dir.joinpath(
        f"{str(input_nifti_path.name).removesuffix('.nii.gz')}_spine_mask.nii.gz"
        )
    
    if spine_mask_path.exists() and not overwrite_output:
        print(f"file '{spine_mask_path}' exists, skipping spine segmentation")
        return {'spine_mask_path': spine_mask_path}
    
    print(f"\nsegmenting vertebrae for {input_nifti_path.name}")
    
    if not vert_classes:
        vert_classes = list(TARGET_VERTEBRAE_MAP.keys())
        print(f"vert_classes is None, using default: {vert_classes}")
    
    start = perf_counter()
    spine_mask: nib.nifti1.Nifti1Image = totalsegmentator(
        input_nifti_path,
        spine_mask_path,                        
        fast=False, 
        ml=True,
        quiet=True,
        task="total",
        roi_subset=vert_classes,
        device="gpu")
    
    duration = perf_counter() - start
    print(f"spine segmentation finised in {duration:.2f} seconds")
    
    spine_mask = nib.funcs.as_closest_canonical(spine_mask)
    
    return {'spine_mask': spine_mask, 'spine_mask_path': spine_mask_path, 'duration': duration}

         
def segment_tissues(
    tissue_volume_path: Union[Path, str],
    case_output_dir: Union[Path, str],
    **kwargs
    ):
    
    if not isinstance(case_output_dir, Path):
        case_output_dir = Path(case_output_dir)
    
    print(f"\nstarting tissue segmentation for {tissue_volume_path.name}")
    filename_parts = tissue_volume_path.name.split("_")[:4] # split results into ["sarco", n, phase, "l3", "slices"]
    output_filename = "_".join(filename_parts) + "_tissue_mask.nii.gz"
    
    output_filepath = Path(case_output_dir, output_filename)

    start = perf_counter()
    predict_cases(
        model=str(MODEL_DIR),
        list_of_lists=[[tissue_volume_path]],
        output_filenames=[output_filepath],
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
    
    duration = perf_counter() - start
    print(f"tissue segmentation finished in {duration}")
    
    mask = nib.as_closest_canonical(nib.load(output_filepath))
    volume = nib.as_closest_canonical(nib.load(tissue_volume_path))
    
    # mask_arr = mask.get_fdata().astype(np.uint8)
    # volume_arr = volume.get_fdata()
    spacing = volume.header.get_zooms()
    # squeeze 2D arrays from H x W x 1 -> H x W    
    # if all((mask_arr.shape[-1], volume_arr.shape[-1])) == 1:
    #     # mask_arr = np.squeeze(mask_arr, axis=-1)
    #     # volume_arr = np.squeeze(volume_arr, axis=-1)
        
    #     # get spacing for first two pixel directions
    #     spacing = volume.header.get_zooms()[:-1]
    # elif all((mask_arr.shape[-1], volume_arr.shape[-1])) > 1:
    #     spacing = volume.header.get_zooms()
    # else:
    #     raise ValueError(f"non matching array shapes between mask {mask_arr.shape} and volume {volume_arr.shape}")
        
    return {
        'volume': volume, 
        'mask': mask, 
        # 'mask_affine': mask.affine, 
        'mask_filepath': output_filepath, 
        'spacing': spacing, 
        'duration': duration
        }
        

def get_vertebrae_body_centroids(
    mask: nib.nifti1.Nifti1Image, 
    vert_labels: int
    ) -> int:
    """
    Get vertebrae's body centroid coordinates in pixel space.

    Args:
        mask (nib.nifti1.Nifti1Image): spine prediction mask
        vert_labels (int): vertebrae mask labels

    Returns:
        body_centroids_z (int): 
        index of vertebrae body centroid along Z (superior) direction in voxel space
    """    
    
    mask_arr = mask.get_fdata().astype(np.uint8)
    
    # get sagittal slice at L3's center
    vert_label_centroid = np.rint(sk.measure.centroid(mask_arr == vert_labels)).astype(np.uint16)
    sagittal_slice_arr = mask_arr[vert_label_centroid[0], ...]
    sagittal_l3 = np.where(sagittal_slice_arr == vert_labels, vert_labels, 0)
    
    # relabel L3 parts
    vert_components = sk.measure.label(sagittal_l3)
    
    # get 2 largest components
    compoments_pixel_num = {prop.label: prop.num_pixels for prop in sk.measure.regionprops(vert_components)}
    largest_components_labels = sorted(compoments_pixel_num, key=compoments_pixel_num.get, reverse=True)
        
    # get centers of largest components and convert from pixel units to physical units
    comp_centroids = np.array([sk.measure.centroid(vert_components == label) for label in largest_components_labels])
    comp_centroids = np.rint(comp_centroids).astype(np.uint16)

    """
    get the centroid in front of whole vertebrae mask centroid
    
    - axis directions/coordinates:
    vertebrae mask centroid: (X, Y, Z) -> (R, A, S)
    centroids: (Y, Z) -> (A, S), centroids numpy shapes (n, 2), where n is number of centroids
    
    - we need to compare centroid coordinates in anterior (A) direction which increases towards anterior
    - comparison is done in pixel space 
    """ 
    vert_body_centroid = comp_centroids[np.argmax(comp_centroids[:, 0] > vert_label_centroid[1])]
    
    return vert_body_centroid[-1]


def extract_slices(
    ct_volume_path: Union[Path, str],
    output_dir: Union[Path, str],
    spine_mask: Union[nib.nifti1.Nifti1Image, Path, str],
    slices_num: int = 0
    ) -> None:

    if any([isinstance(spine_mask, c) for c in (Path, str)]):
        if spine_mask.exists():
            spine_mask = nib.load(spine_mask)
    
    start = perf_counter()
    body_centroids_z = get_vertebrae_body_centroids(spine_mask, TARGET_VERTEBRAE_MAP["vertebrae_L3"])

    ct_volume = nib.load(ct_volume_path)
    ct_volume = nib.funcs.as_closest_canonical(ct_volume)

    if slices_num > 0:
        slices_range = [body_centroids_z - (slices_num // 2), body_centroids_z + (slices_num // 2 ) + 1]
        if slices_range[0] < 0:
            slices_range[0] = 0
            print(f"lower index {slices_range[0]} is outside extent for Z dimension, setting to 0")
        
        z_size = ct_volume.shape[-1]
        if slices_range[1] > z_size:
            slices_range[1] = z_size
            print(f"upper index {slices_range[1]} is outside extent for Z dimension {z_size}, setting to {z_size}")    
    else:
        slices_range = [body_centroids_z, body_centroids_z]
    
    slices_range[1] += 1 # nib slicer requires slice indexes [i:i + 1]
    print(f"extracting {slices_range[1] - slices_range[0]} slices in range {slices_range}, middle slice at index {body_centroids_z}")
    
    sliced_ct_volume = ct_volume.slicer[..., slices_range[0]:slices_range[1]] # nib slicer requires range (i, i + 1) 
    sliced_ct_volume = nib.funcs.as_closest_canonical(sliced_ct_volume)
    
    name = ct_volume_path.name.removesuffix(".nii.gz")
    output_filepath = Path(output_dir, f"{name}_l3_slices.nii.gz")
    if output_filepath.exists():
        print(f"file '{output_filepath}' exists, overwriting")

    try:
        nib.save(sliced_ct_volume, output_filepath)
    except RuntimeError as err:
        print(err)
    
    duration = perf_counter() - start
    print(f"slice extraction finished in {duration} seconds")
    
    return {'sliced_volume_path': output_filepath, 'duration': duration}


def postprocess_tissue_masks(
    mask: nib.Nifti1Image,
    volume: nib.Nifti1Image,
    affine: npt.NDArray = None,
    output_nifti_filepath: Union[Path, str, None] = None
    ):
    
    """
    store tissue labels in separate z-axis
    tissue      | label     | z-axis index
    sat         | 1         | 0
    vat         | 2         | 1
    imat        | 3         | 2
    muscle      | 4         | 3
    """
    start = perf_counter()
    mask_arr = mask.get_fdata().astype(np.uint8)
    volume_arr = volume.get_fdata()
    
    out = np.zeros((*mask_arr.shape, len(TARGET_TISSUES_MAP)), dtype=bool)
    for i, label in enumerate(TARGET_TISSUES_MAP.values()):
        
        # for SAT (label == 1) use 200, for other tissues use 20
        min_hole_size = 200 if label == 1 else 20
            
        out[..., i] = sk.morphology.remove_small_holes(mask_arr == label, min_hole_size)
        
        if label == 4:
            # get muscle tissue pixels in HU
            muscle_hu = volume_arr * out[..., 3]

            imat_hu = np.logical_and(
                muscle_hu <= -30, 
                muscle_hu >= -190
                )

            imat_hu_filt = sk.morphology.remove_small_objects(imat_hu, 10)
            out[imat_hu_filt, 2] = 1
            out[imat_hu_filt, 3] = 0
            
    # convert processed labels into single slice array H x W x L -> H x W
    out_nifti = np.zeros((*out.shape[:-1], 1), dtype=np.uint8)
    for i in range(out.shape[-1]):
        out_nifti[out[..., i] == 1, 0] = i + 1
    out_nifti = nib.as_closest_canonical(nib.Nifti1Image(out_nifti, affine))
    nib.save(out_nifti, output_nifti_filepath) # overwrite segmented image
    
    duration = perf_counter() - start
    print(f"tissue postprocessing finished in {duration:.2f} second")
    return {'processed_masks': out, 'duration': duration}


def compute_metrics(
    tissue_mask_array: npt.NDArray,
    metrics: list,
    spacing=None
    ):
    
    # if None use 1mm spacing in all directions
    if spacing is None:
        if len(tissue_mask_array.shape) == 2:
            spacing = (1., 1.)
        elif len(tissue_mask_array.shape) == 3:
            spacing = (1., 1., 1.)
        else:
            raise ValueError(f"array of shape {tissue_mask_array.shape} is not supported, needs to be (H x W) or (H x W x D)")
    
    
    if len(spacing) == 2:
        pixel_size = np.prod(spacing[:2]) / 100.0 # pixel size is in cm^2
    elif len(spacing) == 3:
        pixel_size = np.prod(spacing) / 10000.0 # pixel size is in cm^3
    
    metric_results = {}
    
    # calculate area in cm^2
    area = {
        tissue: np.count_nonzero(tissue_mask_array[..., TISSUE_LABEL_INDEX.index(tissue)]) * pixel_size
        for tissue in TISSUE_LABEL_INDEX
    }
        
    metric_results['area'] = area
    return metric_results