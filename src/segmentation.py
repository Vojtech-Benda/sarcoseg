import subprocess
from pathlib import Path
from time import perf_counter
from typing import Union

import nibabel as nib
import pandas as pd
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from numpy import nan

from src import slogger, utils, visualization
from src.classes import ImageData, MetricsData, SegmentationResult, StudyData
from src.utils import DEFAULT_VERTEBRA_CLASSES

logger = slogger.get_logger(__name__)

MODEL_DIR = Path("models", "muscle_fat_tissue_stanford_0_0_2")

tissue_predictor = nnUNetPredictor()
tissue_predictor.initialize_from_trained_model_folder(
    "models/mazurowski_muscle_fat", use_folds=(5,)
)


def segment_ct_study(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    slices_num: int = 0,
    save_mask_overlays: bool = False,
    study_case: StudyData = None,
) -> list[MetricsData]:
    if isinstance(output_dir, str):
        Path(output_dir)

    if isinstance(input_dir, str):
        Path(input_dir)

    if not study_case:
        path = input_dir.joinpath(f"dicom_tags_{input_dir.name}.json")
        study_case = utils.read_study_case(path)

    series_nifti_filepaths = list(input_dir.rglob("input_ct_volume.nii.gz"))
    logger.info("-" * 25)
    logger.info(
        f"found {len(series_nifti_filepaths)} volumes to segment spine in directory `{input_dir}`"
    )

    seg_result = SegmentationResult._from_study_case(study_case)

    dict_of_metrics: dict[str, MetricsData] = {}

    for series_filepath in series_nifti_filepaths:
        input_volume_data: ImageData = utils.read_volume(series_filepath)

        series_output_dir = series_filepath.parent
        series_inst_uid = series_output_dir.parts[-1]

        logger.info(
            f"running segmentation on CT series {series_inst_uid} of participant {seg_result.participant}"
        )

        spine_mask_data, spine_duration = segment_spine(
            series_filepath, series_output_dir
        )

        tissue_volume_data, centroids, extraction_duration = utils.extract_slices(
            input_volume_data.image,
            spine_mask_data.image,
            series_output_dir,
            slices_num,
        )

        tissue_mask_data, tissue_duration = segment_tissues(
            tissue_volume_data.path, series_output_dir
        )

        # [TODO] check this and verify if it should be used
        processed_data, postproc_duration = utils.postprocess_tissue_masks(
            tissue_mask_data,
            tissue_volume_data,
        )

        # [TODO] maybe replace skimage with simpleitk?
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

        if save_mask_overlays:
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

    seg_result._write_to_json(output_dir)
    return seg_result


def segment_spine(
    input_nifti_path: Union[str, Path],
    output_dir: Union[str, Path] = None,
    vert_classes: list[str] = None,
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

    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    spine_mask_path = output_dir.joinpath("spine_mask.nii.gz")

    logger.info(f"\nsegmenting vertebrae for {input_nifti_path.name}")

    if not vert_classes:
        vert_classes = list(DEFAULT_VERTEBRA_CLASSES.keys())
        logger.info(f"vert_classes is None, using default: {vert_classes}")

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
    logger.info(f"spine segmentation finised in {duration:.2f} seconds")

    spine_mask = nib.as_closest_canonical(nib.load(spine_mask_path))
    spacing = spine_mask.header.get_zooms()
    return ImageData(image=spine_mask, path=spine_mask_path, spacing=spacing), duration


def segment_tissues(
    tissue_volume_path: Union[Path, str], case_output_dir: Union[Path, str]
) -> tuple[ImageData, float]:
    if isinstance(case_output_dir, str):
        case_output_dir = Path(case_output_dir)

    logger.info(f"\nstarting tissue segmentation for {tissue_volume_path.name}")

    output_filepath = Path(case_output_dir, "tissue_mask.nii.gz")
    print(f"{tissue_volume_path=}")
    start = perf_counter()
    try:
        tissue_predictor.predict_from_files(
            list_of_lists_or_source_folder=[[str(tissue_volume_path)]],
            output_folder_or_list_of_truncated_output_files=[str(output_filepath)],
            num_processes_preprocessing=8,
            num_processes_segmentation_export=8,
        )
    except RuntimeError:
        logger.info(f"nnUNet finished `{tissue_volume_path}`")

    duration = perf_counter() - start
    logger.info(f"tissue segmentation finished in {duration}")

    tissue_mask: ImageData = utils.read_volume(output_filepath)
    return tissue_mask, duration


def write_metric_results(metric_results: list[MetricsData], output_study_dir: Path):
    df = pd.DataFrame([result._to_dict() for result in metric_results])
    filepath = output_study_dir.joinpath(f"metric_results_{output_study_dir.name}.csv")
    if filepath.exists():
        logger.info(f"overwriting existing metric_results.csv at `{filepath}`")
    df.to_csv(filepath, sep=",", na_rep=nan, columns=df.columns, index=None)


# [TODO]: check this and if it needs to be used
# maybe use it as a ArgumentParser command
def collect_all_metric_results(
    input_dir: Union[str, Path], write_to_csv: bool = False
) -> pd.DataFrame:
    if isinstance(input_dir, str):
        input_dir = Path(input_dir)

    metrics_files = list(input_dir.rglob("metric_results_*.csv"))
    df = pd.concat(
        (
            pd.read_csv(file, index_col=None, header=0, dtype={"patient_id": str})
            for file in metrics_files
        ),
        axis=0,
        ignore_index=True,
    )

    logger.info(
        f"collected metric results of {len(df.study_inst_uid.unique())} studies ({len(df.series_inst_uid.unique())} series)"
    )

    if write_to_csv:
        filepath = Path(input_dir, "all_metric_results.csv")
        df.to_csv(filepath, sep=",", na_rep=nan, index=None, columns=df.columns)
        logger.info(f"written results to `{filepath}`")

    return df
