import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from time import perf_counter

import pandas as pd
import SimpleITK as sitk

# from src import slogger
from src.classes import Centroids, ImageData, MetricsData, StudyData

log = logging.getLogger("utils")

DEFAULT_VERTEBRA_CLASSES: dict[str, int] = {
    "vertebrae_L1": 31,
    "vertebrae_L2": 30,
    "vertebrae_L3": 29,
    "vertebrae_L4": 28,
    "vertebrae_L5": 27,
    "vertebrae_S1": 26,
}

DEFAULT_TISSUE_CLASSES: dict[str, int] = {
    "muscle": 1,
    "sat": 2,
    "vat": 3,
    "imat": 4,
}

TISSUE_LABEL_INDEX = list(DEFAULT_TISSUE_CLASSES.keys())

TISSUE_HU_RANGES: dict[str, tuple[int, int]] = {
    "muscle": (-29, 150),
    "imat": (-190, -30),
    "vat": (-205, -51),
}


# logger = slogger.get_logger(__name__)
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

    # check if L3 has been segmented!!
    if l3_label not in label_filt.GetLabels():
        log.warning("no L3 mask label found")
        return Centroids()

    # get the whole L3 vertebrae centroid
    # centroid index = [sagittal, coronal, axial]
    vert_centroid = mask.TransformPhysicalPointToIndex(label_filt.GetCentroid(l3_label))

    # relabel the whole L3 vertebrae in sagittal view
    # label object size sorted descending order
    relabeled_vert_parts = sitk.RelabelComponent(
        sitk.ConnectedComponent(mask[vert_centroid[0], ...] == l3_label),
        sortByObjectSize=True,
    )

    # label of vertebrae body is 1 due to descending sorting by size
    label_filt.Execute(relabeled_vert_parts)
    body_centroid = relabeled_vert_parts.TransformPhysicalPointToIndex(
        label_filt.GetCentroid(1)
    )

    return Centroids(vert_centroid, body_centroid)


def postprocess_tissue_masks(
    mask_data: ImageData,
    volume_data: ImageData,
) -> tuple[ImageData, int | float]:
    start = perf_counter()

    imat_hu_range = TISSUE_HU_RANGES["imat"]
    vat_hu_range = TISSUE_HU_RANGES["vat"]

    # copy the mask for in place modification without affecting original mask
    mask = sitk.Image(mask_data.image)

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


def compute_metrics(
    tissue_mask_data: ImageData,
    tissue_volume_data: ImageData,
    patient_height: float | None = None,
) -> MetricsData:
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
    return MetricsData(area=area, mean_hu=mean_hu, skelet_muscle_index=smi)


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
            sep="[,;\t]",
            engine="python",
        )
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(
            filepath, index_col=False, header=0, dtype=str, usecols=columns
        )

    return df


def read_volume(path: Path | str, orientation: str | None) -> ImageData:
    image = sitk.ReadImage(path)
    if orientation:
        image = sitk.DICOMOrient(image, orientation)
    return ImageData(image, Path(path))


def remove_empty_segmentation_dir(dirpath: str | Path):
    log.debug(f"removing empty segmentation directory `{dirpath}`")
    shutil.rmtree(dirpath)


def remove_dicom_dir(dirpath: str | Path):
    log.debug(f"removing input DICOM directory `{dirpath}`")
    shutil.rmtree(dirpath)


def make_report(
    requested_study_cases: list[StudyData],
    output_dir: Path,
    timestamp: str | None = None,
    verbose: bool = False,
):
    if not timestamp:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    study_dirs = list(output_dir.glob("*"))
    missing_studies = [
        {"participant": pat.participant, "study_instance_uid": pat.study_inst_uid}
        for pat in requested_study_cases
        if output_dir.joinpath(pat.study_inst_uid) not in study_dirs
    ]

    finished_studies = [
        {
            "participant": pat.participant,
            "study_inst_uid": pat.study_inst_uid,
            "preprocessed_count": len(list(study_dir.rglob("input_ct_volume.nii.gz"))),
            "segmentated_count": len(list(study_dir.rglob("tissue_mask.nii.gz"))),
        }
        for pat in requested_study_cases
        if (study_dir := output_dir.joinpath(pat.study_inst_uid)).exists()
    ]

    report = {
        "timestamp": timestamp,
        "output_directory": str(output_dir.resolve()),
        "finished_studies": finished_studies,
        "missing_studies": missing_studies,
    }

    report_path = output_dir.joinpath(f"report_{timestamp}.json")
    with open(report_path, "w") as file:
        json.dump(report, file, indent=4)

    log.debug(f"Segmentation report written to `{report_path}`")
