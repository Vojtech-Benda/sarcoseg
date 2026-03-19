import logging
import subprocess
from pathlib import Path
from time import perf_counter

import SimpleITK as sitk
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from src import utils, visualization
from src.classes import (
    Centroids,
    ImageData,
    ProcessResult,
    SegmentationResult,
    StudyData,
)

# from src.slogger import get_logger
from src.utils import DEFAULT_VERTEBRA_CLASSES

# logger = get_logger(__name__)
log = logging.getLogger("segment")

tissue_predictor = nnUNetPredictor()
tissue_predictor.initialize_from_trained_model_folder(
    "models/mazurowski_muscle_fat", use_folds=(5,)
)


def segment_ct_study(
    input_dir: str | Path,
    output_dir: str | Path,
    study_case: StudyData | None = None,
) -> SegmentationResult:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not study_case:
        study_case = StudyData._from_json(
            input_dir.joinpath(f"dicom_tags_{input_dir.name}.json")
        )

    series_nifti_filepaths = list(input_dir.rglob("input_ct_volume.nii.gz"))
    # logger.info("-" * 25)
    log.debug(
        f"found {len(series_nifti_filepaths)} volumes to segment spine in directory `{input_dir}`"
    )

    seg_result = SegmentationResult._from_study_case(study_case)

    for series_filepath in series_nifti_filepaths:
        series_output_dir = series_filepath.parent
        series_inst_uid = series_output_dir.parts[-1]

        log.info(f"running segmentation series {series_inst_uid}")

        spine_mask_data, spine_duration = segment_spine(
            series_filepath, series_output_dir
        )

        input_volume_data: ImageData = utils.read_volume(series_filepath, "LPI")
        slice_extraction_result = extract_slices(
            input_volume_data.image,
            spine_mask_data.image,
            series_output_dir,
        )

        if not slice_extraction_result:
            log.warning(
                f"participant {seg_result.participant}, study {seg_result.study_inst_uid}, series {series_inst_uid} has no L3 mask"
            )
            seg_result.series_process_result[series_inst_uid] = (
                ProcessResult.MISSING_L3_MASK
            )
            continue

        tissue_volume_data, centroids, extraction_duration = slice_extraction_result

        tissue_mask_data, tissue_duration = segment_tissues(
            tissue_volume_data.path, series_output_dir
        )

        processed_data, postproc_duration = utils.postprocess_tissue_masks(
            tissue_mask_data,
            tissue_volume_data,
        )

        metrics = utils.compute_metrics(
            processed_data,
            tissue_volume_data,
            patient_height=study_case.patient_height,
        )

        metrics.set_duration(
            spine_duration, tissue_duration, extraction_duration, postproc_duration
        )
        metrics.centroids = centroids
        metrics.series_inst_uid = series_inst_uid
        metrics.contrast_phase = study_case.series[series_inst_uid].contrast_phase

        seg_result.metrics_dict[series_inst_uid] = metrics
        seg_result.series_process_result[series_inst_uid] = (
            ProcessResult.SEGMENTATION_FINISHED
        )

        log.debug(f"segmentation finished for series {series_inst_uid}")

        case_images_dir = series_output_dir.joinpath("images")
        case_images_dir.mkdir(exist_ok=True)
        visualization.overlay_spine_mask(
            input_volume_data.image,
            spine_mask_data.image,
            centroids.vertebre_centroid,
            output_dir=case_images_dir,
        )

        visualization.overlay_tissue_mask(
            tissue_volume_data.image,
            processed_data.image,
            output_dir=case_images_dir,
        )
        log.debug("saved mask overlays")

    seg_result._write_to_json(output_dir)
    log.debug(f" {series_inst_uid}")
    return seg_result


def segment_spine(
    input_nifti_path: str | Path,
    output_dir: str | Path = "./",
    vert_classes: list[str] | None = None,
    overwrite_output: bool = False,
) -> tuple[ImageData, float]:
    """
    Segment spine vertebrae. Skips segmentation if a spine_mask.nii.gz file exists.

    Args:
        input_nifti_path (Union[str, Path]): path to nifti file.
        output_dir (Union[str, Path], optional): directory to store segmentation mask.
            Defaults to `"./"`.
        vert_classes (list[str]): vertebrae class names in TotalSegmentator's task `total`.
            For class details see https://github.com/wasserth/TotalSegmentator?tab=readme-ov-file#class-details.

    Returns:
        tuple (ImageData, float): Tuple of segmented spine mask and segmentation duration.
        - **spine_mask** (ImageData): Spine segmentation mask.
        - **duration** (float): Duration of segmentation.
    """

    input_nifti_path = Path(input_nifti_path)
    output_dir = Path(output_dir)

    spine_mask_path = output_dir.joinpath("spine_mask.nii.gz")

    log.debug(f"segmenting vertebrae for {input_nifti_path.name}")

    if not vert_classes:
        vert_classes = list(DEFAULT_VERTEBRA_CLASSES.keys())
        log.debug(f"vert_classes is None, using default: {vert_classes}")

    start = perf_counter()

    command = [
        "TotalSegmentator",
        "-i",
        str(input_nifti_path),
        "-o",
        str(spine_mask_path),
        "--task",
        "total",
        "-ml",
        "-ot",
        "nifti",
        "--fast",
        "--device",
        "gpu",
        "--quiet",
        "--roi_subset",
    ] + list(vert_classes)

    subprocess.run(args=command)

    duration = perf_counter() - start
    log.info(f"spine segmentation finised in {duration:.4f} seconds")

    return utils.read_volume(spine_mask_path, "LPI"), duration


def segment_tissues(
    tissue_volume_path: Path | str, case_output_dir: Path | str
) -> tuple[ImageData, float]:
    tissue_volume_path = Path(tissue_volume_path)
    case_output_dir = Path(case_output_dir)

    log.debug(f"starting tissue segmentation for {tissue_volume_path.name}")

    output_filepath = Path(case_output_dir, "tissue_mask.nii.gz")
    start = perf_counter()
    tissue_predictor.predict_from_files(
        list_of_lists_or_source_folder=[[str(tissue_volume_path)]],
        output_folder_or_list_of_truncated_output_files=[str(output_filepath)],
        num_processes_preprocessing=8,
        num_processes_segmentation_export=8,
    )
    # log.debug(f"nnUNet finished `{tissue_volume_path}`")

    duration = perf_counter() - start
    log.info(f"tissue segmentation finished in {duration:.4f} seconds")

    return utils.read_volume(output_filepath, "LPI"), duration


def extract_slices(
    ct_volume: sitk.Image | Path | str,
    spine_mask: sitk.Image | Path | str,
    output_dir: Path | str,
    slices_num: int = 0,
) -> tuple[ImageData, Centroids, float] | None:
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
        tuple (ImageData, float): Tuple of segmented spine mask and segmentation duration.
        - **tissue_slice** (ImageData): Spine segmentation mask.
        - **centroids** (Centroids): Indexes of whole L3 vertebrae centroid and L3 vertebrae body centroid.
        - **duration** (float): Duration of segmentation.
    """
    if not isinstance(spine_mask, sitk.Image):
        raise TypeError(
            f"spine_mask should be of type sitk.Image, not `{type(spine_mask)}`"
        )

    if not isinstance(ct_volume, sitk.Image):
        raise TypeError(
            f"ct_volume should be of type sitk.Image, not `{type(ct_volume)}`"
        )

    start = perf_counter()
    centroids = utils.get_vertebrae_body_centroids(
        spine_mask, DEFAULT_VERTEBRA_CLASSES["vertebrae_L3"]
    )

    # only need to check for the whole vertebrae centroid
    if not centroids.vertebre_centroid:
        return None

    # keep tissue slice as 3D array to maintain origin etc. relative to input full body CT volume
    tissue_slice = ct_volume[
        ..., centroids.body_centroid[1] : centroids.body_centroid[1] + 1
    ]

    output_filepath = Path(output_dir, "tissue_slices.nii.gz")
    if output_filepath.exists():
        log.debug(f"file `{output_filepath}` exists, overwriting")

    sitk.WriteImage(tissue_slice, output_filepath)

    duration = perf_counter() - start
    log.info(f"slice extraction finished in {duration:.4f} seconds")

    return (
        ImageData(image=tissue_slice, path=output_filepath),
        centroids,
        duration,
    )
