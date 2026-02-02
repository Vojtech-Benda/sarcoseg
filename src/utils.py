import nibabel as nib
import numpy as np
import skimage as sk
import pandas as pd
import os
import shutil

from time import perf_counter
from pathlib import Path
from typing import Union, Any
from nibabel.nifti1 import Nifti1Image
from numpy.typing import NDArray

from src.classes import ImageData, MetricsData, Centroids
from src import slogger

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

logger = slogger.get_logger(__name__)


def get_vertebrae_body_centroids(
    mask: Nifti1Image, vert_labels: int
) -> tuple[NDArray, NDArray]:
    """
    Get vertebrae's body centroid coordinates in pixel space.

    Args:
        mask (nib.nifti1.Nifti1Image): spine prediction mask
        vert_labels (int): vertebrae mask labels

    Returns:
        vert_body_centroid (NDArray):
            Vertebrae body centroid in voxel space.

        vert_centroid (NDArray):
            Vertebrae centroid in voxel space.
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

    # get centers of largest components
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
    
    - compare centroid coordinates in anterior (A) direction which increases towards anterior
    - comparison is done in pixel/voxel space 
    """
    vert_body_centroid = comp_centroids[
        np.argmax(comp_centroids[:, 0] > vert_label_centroid[1])
    ]

    return vert_body_centroid, vert_label_centroid


def extract_slices(
    ct_volume: Union[Nifti1Image, Path, str],
    spine_mask: Union[Nifti1Image, Path, str],
    output_dir: Union[Path, str],
    slices_num: int = 0,
) -> tuple[ImageData, Centroids, float]:
    """
    Extract axial slices from input CT volume at vertebrae body's centroid.

    Args:
        ct_volume (Union[Nifti1Image, Path, str]): Input CT volume.
        spine_mask (Union[Nifti1Image, Path, str]): Segmented spine mask with labeled vertebrae.
        output_dir (Union[Path, str]): directory to save extracted slices data.
        slices_num (int, optional): Number of slices to extract. Defaults to 0. If slicing range extends CT volume size, it will be clipped to volume's size.

    Raises:
        FileNotFoundError: If input CT volume is not found.
        FileNotFoundError: If segmented spine mask is not found.

    Returns:

    """
    if not isinstance(spine_mask, Nifti1Image):
        if Path(spine_mask).exists():
            spine_mask = nib.as_closest_canonical(nib.load(spine_mask))
        else:
            raise FileNotFoundError(spine_mask)

    if not isinstance(ct_volume, Nifti1Image):
        if Path(ct_volume).exists():
            ct_volume = nib.as_closest_canonical(nib.load(ct_volume))
        else:
            raise FileNotFoundError(ct_volume)

    start = perf_counter()
    body_centroid, vert_centroid = get_vertebrae_body_centroids(
        spine_mask, DEFAULT_VERTEBRA_CLASSES["vertebrae_L3"]
    )

    # requires slices_num=2 at minimum
    if slices_num >= 2:
        slices_range = [
            # extract slices in Z direction (superior-inferior)
            body_centroid[-1] - (slices_num // 2),
            body_centroid[-1] + (slices_num // 2),
        ]
    else:
        slices_range = [body_centroid[-1], body_centroid[-1]]

    slices_range[-1] += 1  # nibabel slicer requires range [..., i:i + 1]

    if slices_range[0] < 0:
        slices_range[0] = 0
        logger.info(
            f"lower index {slices_range[0]} is outside lower extent for Z dimension, setting to 0"
        )

    z_size = ct_volume.shape[-1]
    if slices_range[1] > z_size:
        slices_range[1] = z_size
        slices_range[0] -= 1
        logger.info(
            f"upper index {slices_range[1]} is outside upper extent for Z dimension {z_size}, setting to {z_size}"
        )

    logger.info(
        f"extracting {slices_range[1] - slices_range[0]} slices in range {slices_range}"
    )

    sliced_volume = ct_volume.slicer[
        ..., slices_range[0] : slices_range[1]
    ]  # nib slicer requires range [..., i:i + 1]
    sliced_volume = nib.as_closest_canonical(sliced_volume)

    output_filepath = Path(output_dir, "tissue_slices.nii.gz")
    if output_filepath.exists():
        logger.info(f"file `{output_filepath}` exists, overwriting")

    try:
        nib.save(sliced_volume, output_filepath)
    except RuntimeError as err:
        logger.error(err)

    duration = perf_counter() - start
    logger.info(f"slice extraction finished in {duration} seconds")

    return (
        ImageData(image=sliced_volume, path=output_filepath),
        Centroids(vertebre_centroid=vert_centroid, body_centroid=body_centroid),
        duration,
    )


def postprocess_tissue_masks(
    mask_data: ImageData,
    volume_data: ImageData,
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
    mask_arr = mask_data.image.get_fdata().astype(np.uint8)
    volume_arr = volume_data.image.get_fdata()

    processed_mask = np.zeros(
        (*mask_arr.shape, len(DEFAULT_TISSUE_CLASSES)), dtype=bool
    )
    for i, label in enumerate(DEFAULT_TISSUE_CLASSES.values()):
        # for SAT (label == 1) use 200, for other tissues use 20
        min_hole_size = 200 if label == 1 else 20

        processed_mask[..., i] = sk.morphology.remove_small_holes(
            mask_arr == label, min_hole_size
        )

        if label == 4:
            # get muscle tissue pixels in HU
            muscle_hu = volume_arr * processed_mask[..., 3]

            imat_hu = np.logical_and(muscle_hu <= -30, muscle_hu >= -190)

            imat_hu_filt = sk.morphology.remove_small_objects(imat_hu, 10)
            processed_mask[imat_hu_filt, 2] = 1
            processed_mask[imat_hu_filt, 3] = 0

    # squeeze processed labels of shape H x W x D x L -> H x W x D
    out_nifti = np.zeros(processed_mask.shape[:-1], dtype=np.uint8)

    for i, label in enumerate(DEFAULT_TISSUE_CLASSES.values()):
        out_nifti[processed_mask[..., i]] = label
    out_nifti = nib.as_closest_canonical(
        nib.Nifti1Image(out_nifti, mask_data.image.affine)
    )
    processed_mask_path = mask_data.path.parent.joinpath("tissue_mask_pp.nii.gz")
    nib.save(out_nifti, processed_mask_path)  # overwrite segmented image

    duration = perf_counter() - start
    logger.info(f"tissue postprocessing finished in {duration:.2f} second")
    return ImageData(
        image=out_nifti, spacing=mask_data.spacing, path=processed_mask_path
    ), duration


def compute_metrics(
    tissue_mask_data: ImageData,
    tissue_volume_data: ImageData,
    series_df_tags: dict[str, Any] = None,
):
    if tissue_mask_data.spacing:
        spacing = tissue_mask_data.spacing
    elif tissue_volume_data.spacing:
        spacing = tissue_volume_data.spacing

    # if None use 1mm spacing in all directions
    if spacing is None:
        spacing = (1.0, 1.0, 1.0)

    mask_arr = tissue_mask_data.image.get_fdata()
    # if len(mask_arr.shape[:-1]) != len(spacing):
    #     raise ValueError(
    #         f"array shape {mask_arr.shape} does not match spacing shape {spacing}, needs to be (H x W x 1) or (H x W x D)"
    #     )

    depth = mask_arr.shape[-1]  # get image size only
    if depth == 1:
        pixel_size = np.prod(spacing[:2]) / 100.0  # pixel size is in cm^2
    elif depth > 1:
        pixel_size = np.prod(spacing) / 10000.0  # pixel size is in cm^3

    tissue_arr = tissue_volume_data.image.get_fdata()

    metrics_data = MetricsData()
    metrics_data.area = {
        tissue: np.count_nonzero(mask_arr == label) * pixel_size
        for tissue, label in DEFAULT_TISSUE_CLASSES.items()
    }
    metrics_data.mean_hu = {
        tissue: np.mean(tissue_arr[mask_arr == label])
        for tissue, label in DEFAULT_TISSUE_CLASSES.items()
    }

    patient_height = series_df_tags.get("vyska_pac.", None)
    if patient_height:
        # skeletal muscle index (cm^2 / m^2) = muscle area (cm^2) / patient height (m^2)
        metrics_data.skelet_muscle_index = metrics_data.area["muscle"] / (
            (patient_height / 100.0) ** 2
        )
    return metrics_data


def read_patient_list(filepath: Union[str, Path]) -> Union[dict, None]:
    if isinstance(filepath, str):
        filepath = Path(filepath)

    if not filepath.is_file():
        logger.error(f"patient list at `{filepath}` is not a file")
        return None
    if not filepath.exists():
        logger.error(f"patient list at `{filepath}` not found")
        return None

    suffix = filepath.suffix
    if suffix == ".csv":
        df = pd.read_csv(
            filepath, index_col=False, header=0, dtype=str, usecols=["participant"]
        )
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(
            filepath, index_col=False, header=0, dtype=str, usecols=["participant"]
        )

    return df.participant.to_list()


def get_series_tags(df_tags: pd.DataFrame, series_inst_uid: str):
    return df_tags.loc[df_tags["series_inst_uid"] == series_inst_uid].iloc[0].to_dict()


def read_volume(path: Union[Path, str]):
    volume = nib.as_closest_canonical(nib.load(path))
    spacing = volume.header.get_zooms()
    return ImageData(image=volume, spacing=spacing, path=path)


def remove_empty_segmentation_dir(dirpath: Union[str, Path]):
    logger.info(f"removing empty segmentation directory `{dirpath}`")
    shutil.rmtree(dirpath)


def remove_dicom_dir(dirpath: Union[str, Path]):
    logger.info(f"removing input DICOM directory `{dirpath}`")
    shutil.rmtree(dirpath)
