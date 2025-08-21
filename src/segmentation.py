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
from src import visualization


DEFAULT_VERTEBRA_CLASSES = {
    "vertebrae_L1": 31,
    "vertebrae_L2": 30,
    "vertebrae_L3": 29,
    "vertebrae_L4": 28,
    "vertebrae_L5": 27,
    "vertebrae_S1": 26,
}


DEFAULT_TISSUE_CLASSES = {"sat": 1, "vat": 2, "imat": 3, "muscle": 4}

TISSUE_LABEL_INDEX = list(DEFAULT_TISSUE_CLASSES.keys())

MODEL_DIR = Path("models", "muscle_fat_tissue_stanford_0_0_2")


def segment_ct(input_dir: str, output_dir: str, additional_metrics: list, **kwargs):
    case_dirs: list[Path] = list(Path(input_dir).glob("*/"))
    print(f"found {len(case_dirs)} cases")

    for case_dir in case_dirs:
        case_nifti_dir = Path(output_dir, case_dir.name, "nifti")
        case_images_dir = Path(output_dir, case_dir.name, "images")
        case_nifti_dir.mkdir(exist_ok=True, parents=True)
        case_images_dir.mkdir(exist_ok=True, parents=True)

        ct_volume_paths = list(case_dir.glob("*.nii.gz"))
        print("\n" + "-" * 25)
        print(
            f"\nfound {len(ct_volume_paths)} volumes to segment spine for case {case_dir.name}"
        )

        slices_num = kwargs.get("slices_num", 0)
        save_segmentations = kwargs.get("save_segmentations")

        for ct_volume_path in ct_volume_paths:
            spine_results = segmentation.segment_spine(
                ct_volume_path,
                case_nifti_dir,
            )

            spine_mask = (
                spine_results["spine_mask"]
                if "spine_mask" in spine_results
                else spine_results["spine_mask_path"]
            )

            slice_results = segmentation.extract_slices(
                ct_volume_path, case_nifti_dir, spine_mask, slices_num
            )

            tissue_results = segmentation.segment_tissues(
                slice_results["sliced_volume_path"], case_nifti_dir
            )

            postproc_results = segmentation.postprocess_tissue_masks(
                tissue_results["mask"],
                tissue_results["volume"],
                tissue_results["mask_filepath"],
            )

            metric_results = segmentation.compute_metrics(
                postproc_results["processed_mask"],
                metrics=additional_metrics,
                spacing=tissue_results["spacing"],
            )

            phase = str(ct_volume_path.name).removesuffix(".nii.gz")
            visualization.overlay_spine_mask(
                ct_volume_path,
                spine_results["spine_mask_path"],
                slice_results["vert_centroid"],
                output_dir=case_images_dir,
                phase=phase,
            )

            visualization.overlay_tissue_mask(
                slice_results["sliced_volume_path"],
                tissue_results["mask_filepath"],
                output_dir=case_images_dir,
                phase=phase,
            )


def segment_spine(
    input_nifti_path: Union[str, Path],
    output_dir: Union[str, Path] = None,
    vert_classes: list = None,
    overwrite_output: bool = False,
) -> dict:
    """
    Segment spine vertebrae.

    Args:
        input_nifti_path (Union[str, Path]): path to nifti file
        output_dir (Union[str, Path], optional): directory to store segmentation mask. Defaults to "./".

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
        return {"spine_mask_path": spine_mask_path}

    print(f"\nsegmenting vertebrae for {input_nifti_path.name}")

    if not vert_classes:
        vert_classes = list(DEFAULT_VERTEBRA_CLASSES.keys())
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
        device="gpu",
    )

    duration = perf_counter() - start
    print(f"spine segmentation finised in {duration:.2f} seconds")

    spine_mask = nib.funcs.as_closest_canonical(spine_mask)

    return {
        "spine_mask": spine_mask,
        "spine_mask_path": spine_mask_path,
        "duration": duration,
    }


def segment_tissues(
    tissue_volume_path: Union[Path, str], case_output_dir: Union[Path, str], **kwargs
):
    if not isinstance(case_output_dir, Path):
        case_output_dir = Path(case_output_dir)

    print(f"\nstarting tissue segmentation for {tissue_volume_path.name}")

    # split results into [phase, "slices"]
    output_filename = tissue_volume_path.name.removesuffix(".nii.gz") + "_mask.nii.gz"

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
        disable_postprocessing=True,
    )

    duration = perf_counter() - start
    print(f"tissue segmentation finished in {duration}")

    mask = nib.as_closest_canonical(nib.load(output_filepath))
    volume = nib.as_closest_canonical(nib.load(tissue_volume_path))
    spacing = volume.header.get_zooms()

    return {
        "volume": volume,
        "mask": mask,
        "mask_filepath": output_filepath,
        "spacing": spacing,
        "duration": duration,
    }


def get_vertebrae_body_centroids(
    mask: nib.nifti1.Nifti1Image, vert_labels: int
) -> npt.NDArray:
    """
    Get vertebrae's body centroid coordinates in pixel space.

    Args:
        mask (nib.nifti1.Nifti1Image): spine prediction mask
        vert_labels (int): vertebrae mask labels

    Returns:
        body_centroid (npt.NDArray):
        vertebrae body centroid in voxel space
    """

    mask_arr = mask.get_fdata().astype(np.uint8)

    # get sagittal slice at L3's center
    vert_label_centroid = np.rint(sk.measure.centroid(mask_arr == vert_labels)).astype(
        np.uint16
    )
    sagittal_slice_arr = mask_arr[vert_label_centroid[0], ...]
    sagittal_l3 = np.where(sagittal_slice_arr == vert_labels, vert_labels, 0)

    # relabel L3 parts
    vert_components = sk.measure.label(sagittal_l3)

    # get 2 largest components
    compoments_pixel_num = {
        prop.label: prop.num_pixels for prop in sk.measure.regionprops(vert_components)
    }
    largest_components_labels = sorted(
        compoments_pixel_num, key=compoments_pixel_num.get, reverse=True
    )

    # get centers of largest components and convert from pixel units to physical units
    comp_centroids = np.array(
        [
            sk.measure.centroid(vert_components == label)
            for label in largest_components_labels
        ]
    )
    comp_centroids = np.rint(comp_centroids).astype(np.uint16)

    """
    get the centroid in front of whole vertebrae mask centroid
    
    - axis directions/coordinates:
    vertebrae mask centroid: (X, Y, Z) -> (R, A, S)
    centroids: (Y, Z) -> (A, S), centroids numpy shapes (n, 2), where n is number of centroids
    
    - we need to compare centroid coordinates in anterior (A) direction which increases towards anterior
    - comparison is done in pixel space 
    """
    vert_body_centroid = comp_centroids[
        np.argmax(comp_centroids[:, 0] > vert_label_centroid[1])
    ]

    return vert_body_centroid, vert_label_centroid


def extract_slices(
    ct_volume_path: Union[Path, str],
    output_dir: Union[Path, str],
    spine_mask: Union[nib.nifti1.Nifti1Image, Path, str],
    slices_num: int = 0,
) -> dict:
    if any([isinstance(spine_mask, c) for c in (Path, str)]):
        if spine_mask.exists():
            spine_mask = nib.load(spine_mask)

    start = perf_counter()
    body_centroid, vert_centroid = get_vertebrae_body_centroids(
        spine_mask, DEFAULT_VERTEBRA_CLASSES["vertebrae_L3"]
    )

    ct_volume = nib.load(ct_volume_path)
    ct_volume = nib.funcs.as_closest_canonical(ct_volume)

    # requires slices_num=2 at minimum
    if slices_num > 1:
        slices_range = [
            # extract slices in Z direction (superior-inferior)
            body_centroid[-1] - (slices_num // 2),
            body_centroid[-1] + (slices_num // 2),
        ]
        if slices_range[0] < 0:
            slices_range[0] = 0
            print(
                f"lower index {slices_range[0]} is outside extent for Z dimension, setting to 0"
            )

        z_size = ct_volume.shape[-1]
        if slices_range[1] > z_size:
            slices_range[1] = z_size
            print(
                f"upper index {slices_range[1]} is outside extent for Z dimension {z_size}, setting to {z_size}"
            )
    else:
        slices_range = [body_centroid[-1], body_centroid[-1]]

    # slices_range[1] += 1  # nib slicer requires slice indexes [i:i + 1]
    print(
        f"extracting {slices_range[1] - slices_range[0]} slices in range {slices_range}, middle slice at index {body_centroid}"
    )

    sliced_ct_volume = ct_volume.slicer[
        ..., slices_range[0] : slices_range[1] + 1
    ]  # nib slicer requires range [..., i:i + 1]
    sliced_ct_volume = nib.funcs.as_closest_canonical(sliced_ct_volume)

    name = ct_volume_path.name.removesuffix(".nii.gz")
    output_filepath = Path(output_dir, f"{name}_tissue.nii.gz")
    if output_filepath.exists():
        print(f"file '{output_filepath}' exists, overwriting")

    try:
        nib.save(sliced_ct_volume, output_filepath)
    except RuntimeError as err:
        print(err)

    duration = perf_counter() - start
    print(f"slice extraction finished in {duration} seconds")

    return {
        "sliced_volume_path": output_filepath,
        "duration": duration,
        "body_centroid": body_centroid,
        "vert_centroid": vert_centroid,
    }


def postprocess_tissue_masks(
    mask: nib.Nifti1Image,
    volume: nib.Nifti1Image,
    output_nifti_filepath: Union[Path, str, None] = None,
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

    out = np.zeros((*mask_arr.shape, len(DEFAULT_TISSUE_CLASSES)), dtype=bool)
    for i, label in enumerate(DEFAULT_TISSUE_CLASSES.values()):
        # for SAT (label == 1) use 200, for other tissues use 20
        min_hole_size = 200 if label == 1 else 20

        out[..., i] = sk.morphology.remove_small_holes(mask_arr == label, min_hole_size)

        if label == 4:
            # get muscle tissue pixels in HU
            muscle_hu = volume_arr * out[..., 3]

            imat_hu = np.logical_and(muscle_hu <= -30, muscle_hu >= -190)

            imat_hu_filt = sk.morphology.remove_small_objects(imat_hu, 10)
            out[imat_hu_filt, 2] = 1
            out[imat_hu_filt, 3] = 0

    # squeeze processed labels of shape H x W x D x L -> H x W x D
    out_nifti = np.zeros(out.shape[:-1], dtype=np.uint8)

    for i in range(out.shape[-1]):
        out_nifti[out[..., i] == 1] = i + 1
    out_nifti = nib.as_closest_canonical(nib.Nifti1Image(out_nifti, mask.affine))
    nib.save(out_nifti, output_nifti_filepath)  # overwrite segmented image

    duration = perf_counter() - start
    print(f"tissue postprocessing finished in {duration:.2f} second")
    return {"processed_mask": out, "duration": duration}


def compute_metrics(tissue_mask_array: npt.NDArray, metrics: list, spacing=None):
    # if None use 1mm spacing in all directions
    if spacing is None:
        spacing = (1.0, 1.0, 1.0)

    if len(tissue_mask_array.shape[:-1]) != len(spacing):
        raise ValueError(
            f"array shape {tissue_mask_array.shape} does not match spacing shape {spacing}, needs to be (H x W x 1) or (H x W x D)"
        )

    mask_shape = tissue_mask_array.shape[:-1]  # get image size only
    if mask_shape[-1] == 1:
        pixel_size = np.prod(spacing[:2]) / 100.0  # pixel size is in cm^2
    elif mask_shape[-1] > 1:
        pixel_size = np.prod(spacing) / 10000.0  # pixel size is in cm^3

    metric_results = {}

    # calculate area in cm^2
    area = {
        tissue: np.count_nonzero(
            tissue_mask_array[..., TISSUE_LABEL_INDEX.index(tissue)]
        )
        * pixel_size
        for tissue in TISSUE_LABEL_INDEX
    }
    mean_hu = {
        tissue: np.mean(tissue_mask_array[..., TISSUE_LABEL_INDEX.index(tissue)])
        for tissue in TISSUE_LABEL_INDEX
    }

    metric_results["area"] = area
    metric_results["mean_hu"] = mean_hu
    return metric_results
