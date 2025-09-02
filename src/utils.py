import nibabel as nib
import numpy as np
import skimage as sk

from time import perf_counter
from pathlib import Path
from typing import Union
from nibabel.nifti1 import Nifti1Image
from numpy.typing import NDArray


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


def get_vertebrae_body_centroids(mask: Nifti1Image, vert_labels: int) -> NDArray:
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
    spine_mask: Union[Nifti1Image, Path, str],
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

    # name = ct_volume_path.name.removesuffix(".nii.gz")
    output_filepath = Path(output_dir, "tissue_slices.nii.gz")
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
    mask: Nifti1Image,
    volume: Nifti1Image,
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


def compute_metrics(tissue_mask_array: NDArray, metrics: list, spacing=None):
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
