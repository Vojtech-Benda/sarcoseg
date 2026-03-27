import logging
import shutil
from pathlib import Path
from time import perf_counter

# import nibabel as nib
# import numpy as np
import pandas as pd

# import skimage as sk
import SimpleITK as sitk

from src.classes import Centroids, ImageData, Metrics
from src.labels import (
    DEFAULT_TISSUE_CLASSES,
    # DEFAULT_TISSUE_CLASSES_INV,
    TISSUE_HU_RANGES,
)

log = logging.getLogger("utils")


def get_vertebrae_body_centroids(mask: sitk.Image, l3_label: int) -> Centroids:
    """
    Get vertebrae's body centroid coordinates in pixel space.

    Args:
        mask (sitk.Image): spine prediction mask
        vert_labels (int): vertebrae mask labels

    Returns:
        vert_body_centroid (list[int]):
            Vertebrae body centroid in voxel space.

        vert_centroid (list[int]):
            Vertebrae centroid in voxel space.
    """

    label_filt = sitk.LabelShapeStatisticsImageFilter()
    label_filt.Execute(mask)

    # check if L3 mask is present!!
    if l3_label not in label_filt.GetLabels():
        log.warning("no L3 mask label found")
        return Centroids()

    # get the whole L3 vertebrae centroid
    # centroid index = [sagittal, coronal, axial]
    vert_centroid = mask.TransformPhysicalPointToIndex(label_filt.GetCentroid(l3_label))

    relabeled_vert_parts = sitk.RelabelComponent(
        sitk.ConnectedComponent(mask[vert_centroid[0], ...] == l3_label),
        sortByObjectSize=True,
    )

    # label of vertebrae body is 1 due to descending sort by size
    label_filt.Execute(relabeled_vert_parts)
    body_centroid = relabeled_vert_parts.TransformPhysicalPointToIndex(
        label_filt.GetCentroid(1)
    )

    return Centroids(vert_centroid, body_centroid)

    # vert_centroid = np.rint(sk.measure.centroid(l3_mask)).astype(np.uint16)
    # relabeled_vert_parts = sk.measure.label(l3_mask[vert_centroid[0], ...])

    # # L3 vertebrae body label should always be bigger (more number of pixels)
    # # than L3 spinal process label in sagittal view at L3 whole vertebrae center
    # label_props = sorted(
    #     sk.measure.regionprops(relabeled_vert_parts),
    #     key=lambda x: x["num_pixels"],
    #     reverse=True,
    # )
    # body_centroid = np.rint(label_props[0]["centroid"]).astype(np.uint16)
    # return Centroids(vert_centroid.tolist(), body_centroid.tolist())


def postprocess_tissue_masks(
    mask_data: ImageData,
    volume_data: ImageData,
) -> tuple[ImageData, int | float]:
    start = perf_counter()

    imat_hu_range = TISSUE_HU_RANGES["imat"]
    vat_hu_range = TISSUE_HU_RANGES["vat"]

    # copy the mask for in place modification without affecting original mask

    mask = sitk.Image(mask_data.image)

    if not (mask.GetOrigin == volume_data.image.GetOrigin()):
        log.warning(
            "mask and tissue images do not occupy same physical space, resampling..."
        )

        mask = sitk.Resample(
            mask,
            referenceImage=volume_data.image,
            interpolator=sitk.sitkNearestNeighbor,
            defaultPixelValue=0,
            outputPixelType=mask.GetPixelID(),
        )

    # tissue_arr = volume_data.image.get_fdata()
    # mask_arr = mask_data.image.get_fdata().copy()

    imat_thresh = (imat_hu_range[0] <= volume_data.image <= imat_hu_range[1]) & (
        mask == DEFAULT_TISSUE_CLASSES["muscle"]
    )
    imat_thresh = sitk.BinaryOpeningByReconstruction(imat_thresh)
    mask[imat_thresh] = DEFAULT_TISSUE_CLASSES["imat"]

    non_vat_thresh = (volume_data.image > vat_hu_range[1]) * (
        mask == DEFAULT_TISSUE_CLASSES["vat"]
    )
    non_vat_thresh = sitk.BinaryOpeningByReconstruction(non_vat_thresh)
    mask[non_vat_thresh] = 0

    processed_mask_path = mask_data.path.parent.joinpath("tissue_mask_pp.nii.gz")
    sitk.WriteImage(mask, processed_mask_path)

    duration = perf_counter() - start
    log.info(f"tissue postprocessing finished in {duration:.4f} second")
    return ImageData(image=mask, path=processed_mask_path), duration

    # imat_thresh = np.logical_and(
    #     tissue_arr >= imat_hu_range[0], tissue_arr <= imat_hu_range[1]
    # ) & (mask_arr == DEFAULT_TISSUE_CLASSES["muscle"])

    # footprint = sk.morphology.ball(1)

    # imat_thresh = sk.morphology.remove_small_objects(
    #     sk.morphology.opening(imat_thresh, footprint), max_size=6
    # )
    # mask_arr[imat_thresh] = DEFAULT_TISSUE_CLASSES["imat"]

    # non_vat_thresh = (tissue_arr > vat_hu_range[1]) * (
    #     mask_arr == DEFAULT_TISSUE_CLASSES["vat"]
    # )
    # non_vat_thresh = sk.morphology.remove_small_objects(
    #     sk.morphology.opening(non_vat_thresh, footprint), max_size=6
    # )
    # mask_arr[non_vat_thresh] = 0

    # processed_mask_path = mask_data.path.parent.joinpath("tissue_mask_pp.nii.gz")

    # processed_mask = nib.Nifti1Image(
    #     mask_arr, affine=mask_data.image.affine, dtype=mask_arr.dtype
    # )
    # nib.save(processed_mask, processed_mask_path)

    # duration = perf_counter() - start
    # log.info(f"tissue postprocessing finished in {duration:.4f} second")
    # return ImageData(image=processed_mask, path=processed_mask_path), duration


def compute_metrics(
    tissue_mask_data: ImageData,
    tissue_volume_data: ImageData,
    patient_height: float | None = None,
) -> Metrics:
    """Compute area and mean Hounsfield Unit for segmented tissue masks.
    Also compute skeletal muscle index (SMI) if `patient_height` is given. Patient height needs to be in cm.

    Units of computed metrics:
        - area: cm^2
        - mean_hu: HU
        - SMI: cm^2 / m^2

    Args:
        tissue_mask_data (ImageData): Segmented tissue masks of SAT, VAT, IMAT and MUSCLE.
        tissue_volume_data (ImageData): Input nifti volume.
        patient_height (float | None, optional): Patient's height. Defaults to None.

    Returns:
        metrics (MetricsData): Computed metrics with cross-sectional area, mean HU and skeletal muscle index.
    """
    mask_image = tissue_mask_data.image
    tissue_image = tissue_volume_data.image

    if mask_image.GetSize()[-1] == 1:
        # 3D array to 2D array only for metrics
        mask_image = mask_image[..., 0]
        tissue_image = tissue_image[..., 0]

    stats_filt = sitk.LabelIntensityStatisticsImageFilter()
    stats_filt.Execute(mask_image, tissue_image)

    # divide by 100 to convert from mm2 to cm2
    area = {
        tissue: stats_filt.GetPhysicalSize(label) / 100.0
        for tissue, label in DEFAULT_TISSUE_CLASSES.items()
    }
    mean_hu = {
        tissue: stats_filt.GetMean(label)
        for tissue, label in DEFAULT_TISSUE_CLASSES.items()
    }

    smi = 0.0
    if patient_height:
        # skeletal muscle index (smi) = muscle area / (patient height ^ 2)
        # units: cm2 / m2 = (cm2) / (cm / 100) ^ 2
        smi = area["muscle"] / ((patient_height / 100.0) ** 2)
    return Metrics(area=area, mean_hu=mean_hu, skelet_muscle_index=smi)

    # mask_arr = tissue_mask_data.image.get_fdata().astype(np.uint8)
    # tissue_arr = tissue_volume_data.image.get_fdata()
    # spacing = np.array(tissue_mask_data.image.header.get_zooms())

    # if len(mask_arr.shape) == 3 and mask_arr.shape[-1] == 1:
    #     # 3D array to 2D array only for metrics
    #     mask_arr = mask_arr[..., 0]
    #     tissue_arr = tissue_arr[..., 0]
    #     spacing = spacing[:-1]

    # props = sk.measure.regionprops(mask_arr, tissue_arr, spacing=spacing)

    # # divide by 100 to convert from mm2 to cm2
    # area = {DEFAULT_TISSUE_CLASSES_INV[p.label]: p.area / 100.0 for p in props}
    # mean_hu = {DEFAULT_TISSUE_CLASSES_INV[p.label]: p.intensity_mean for p in props}

    # smi = 0.0
    # if patient_height:
    #     # skeletal muscle index (smi) = muscle area / (patient height ^ 2)
    #     # units: cm2 / m2 = (cm2) / (cm / 100) ^ 2
    #     smi = area["muscle"] / ((patient_height / 100.0) ** 2)
    # return Metrics(
    #     area=area,
    #     mean_hu=mean_hu,
    #     skelet_muscle_index=smi,
    # )


def read_patient_list(
    filepath: str | Path, columns: list[str] | None = None
) -> pd.DataFrame:
    if isinstance(filepath, str):
        filepath = Path(filepath)

    if not filepath.is_file() or not filepath.exists():
        # log.error(f"patient list at `{filepath}` is not a file or doesn't exist")
        raise FileNotFoundError(f"Patient list file not found at {filepath}")

    suffix = filepath.suffix
    if suffix == ".csv":
        df = pd.read_csv(
            filepath,
            index_col=False,
            header=0,
            dtype=str,
            usecols=columns,
            sep=";",
            engine="python",
        )
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(
            filepath, index_col=False, header=0, dtype=str, usecols=columns
        )

    return df


def read_volume(path: Path | str, orientation: str | None = "LPS") -> ImageData:
    image = sitk.ReadImage(path)
    if orientation:
        image = sitk.DICOMOrient(image, orientation)
    return ImageData(image, Path(path))
    # image = nib.load(path)
    # if orientation:
    #     log.debug(
    #         f"transforming loaded image orientation from {nib.aff2axcodes(image.affine)} into {orientation} orientation"
    #     )
    #     if orientation == "RAS":
    #         image = nib.as_closest_canonical(image)
    #     else:
    #         orig_ornt = nib.io_orientation(image.affine)
    #         transform = nib.orientations.ornt_transform(
    #             orig_ornt, nib.orientations.axcodes2ornt(orientation)
    #         )
    #         image = image.as_reoriented(transform)
    # return ImageData(image, Path(path))


def remove_empty_segmentation_dir(dirpath: str | Path):
    log.debug(f"removing empty segmentation directory `{dirpath}`")
    shutil.rmtree(dirpath)


def remove_dicom_dir(dirpath: str | Path):
    log.debug(f"removing input DICOM directory `{dirpath}`")
    shutil.rmtree(dirpath)
