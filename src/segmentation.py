from typing import Union
from pathlib import Path
from time import perf_counter
from datetime import datetime
import shutil
import pandas as pd
from totalsegmentator.python_api import totalsegmentator

import nibabel as nib
from numpy import nan
from nnunet.inference.predict import predict_cases

from src import visualization
from src import utils
from src.utils import DEFAULT_VERTEBRA_CLASSES
from src.classes import ImageData, MetricsData

MODEL_DIR = Path("models", "muscle_fat_tissue_stanford_0_0_2")


def segment_ct_study(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    slices_num: int = 0,
    save_mask_overlays: bool = False,
    collect_metric_results: bool = False,
) -> list[MetricsData]:
    # case_dirs = list(Path(input_dir).glob("*/"))
    # print(f"found {len(case_dirs)} case directories")

    # output_dir = Path(output_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    # output_dir.mkdir()
    # for case_dir in case_dirs:
    series_nifti_files = list(input_dir.rglob("*.nii.gz"))

    usecols = [
        "participant",
        "patient_id",
        "study_inst_uid",
        "series_inst_uid",
        "contrast_phase",
        "vyska_pac.",
    ]
    df_tags = pd.read_csv(
        Path(input_dir, "dicom_tags.csv"),
        index_col=False,
        header=0,
        usecols=lambda col: col in usecols,
        dtype={"patient_id": str, "vyska_pac.": float},
    )

    print("-" * 25)
    print(
        f"\nfound {len(series_nifti_files)} volumes to segment spine for case `{input_dir.name}`"
    )

    metric_results_list: list[MetricsData] = []

    for series_file in series_nifti_files:
        # FIXME: POSSIBLY REMOVE LINES BELOW
        # input_volume.nii.gz already present in output dir
        # just construct the series directory as destination
        # from path [input_dir, ..., study_inst_uid, series_inst_uid, file] take study_inst_uid and series_inst_uid
        case_output_dir = Path(output_dir, *series_file.parent.parts[-2:])
        case_output_dir.mkdir(exist_ok=True, parents=True)

        # FIXME: REMOVE LINE BELOW
        # no need to copy, input_volume.nii.gz already present in output dir
        # shutil.copy2(series_file, case_output_dir.joinpath(series_file.name))

        input_volume_data: ImageData = utils.read_volume(series_file)

        series_inst_uid = case_output_dir.parts[-1]
        print(f"running segmentation on `{series_inst_uid}`")
        spine_mask_data, spine_duration = segment_spine(series_file, case_output_dir)

        tissue_volume_data, centroids, extraction_duration = utils.extract_slices(
            input_volume_data.image,
            spine_mask_data.image,
            case_output_dir,
            slices_num,
        )

        tissue_mask_data, tissue_duration = segment_tissues(
            tissue_volume_data.path, case_output_dir
        )

        processed_data, postproc_duration = utils.postprocess_tissue_masks(
            tissue_mask_data,
            tissue_volume_data,
        )

        series_tags = utils.get_series_tags(df_tags, series_inst_uid)
        metrics_results = utils.compute_metrics(
            processed_data,
            tissue_volume_data,
            series_df_tags=series_tags,
        )

        metrics_results.set_patient_data(df_tags, series_inst_uid)
        metrics_results.set_duration(
            spine_duration, tissue_duration, extraction_duration, postproc_duration
        )
        metrics_results.centroids = centroids
        metric_results_list.append(metrics_results)

        if save_mask_overlays:
            case_images_dir = case_output_dir.joinpath("images")
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

    write_metric_results(
        metric_results_list, Path(output_dir, case_output_dir.parts[-2])
    )

    if collect_metric_results:
        collect_all_metric_results(output_dir)

    return metric_results_list


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
            Defaults to "./".
        vert_classes (list[str]): vertebrae class names in TotalSegmentator's task `total`.
            For class details see https://github.com/wasserth/TotalSegmentator?tab=readme-ov-file#class-details.

    Returns:
        spine_mask (nib.Nifti1Image):
            Spine segmentation mask.

        spine_mask_path (Path):
            Path to spine segmentation mask nifti file.

        duration (float):
            Segmentation runtime, defaults to 0.0 if segmentation file exists.
    """

    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    spine_mask_path = output_dir.joinpath("spine_mask.nii.gz")

    print(f"\nsegmenting vertebrae for {input_nifti_path.name}")

    if not vert_classes:
        vert_classes = list(DEFAULT_VERTEBRA_CLASSES.keys())
        print(f"vert_classes is None, using default: {vert_classes}")

    start = perf_counter()
    spine_mask: nib.Nifti1Image = totalsegmentator(
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

    spine_mask = nib.as_closest_canonical(spine_mask)
    spacing = spine_mask.header.get_zooms()
    return ImageData(image=spine_mask, path=spine_mask_path, spacing=spacing), duration


def segment_tissues(
    tissue_volume_path: Union[Path, str], case_output_dir: Union[Path, str]
) -> tuple[ImageData, float]:
    if isinstance(case_output_dir, str):
        case_output_dir = Path(case_output_dir)

    print(f"\nstarting tissue segmentation for {tissue_volume_path.name}")

    output_filepath = Path(case_output_dir, "tissue_mask.nii.gz")

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

    tissue_mask: ImageData = utils.read_volume(output_filepath)
    return tissue_mask, duration


def write_metric_results(metric_results: list[MetricsData], output_study_dir: Path):
    df = pd.DataFrame([result._to_dict() for result in metric_results])
    filepath = output_study_dir.joinpath("metric_results.csv")
    if filepath.exists():
        print(f"overwriting existing metric_results.csv at `{filepath}`")
    df.to_csv(filepath, sep=",", na_rep=nan, columns=df.columns, index=None)


def collect_all_metric_results(
    input_dir: Union[str, Path], write_to_csv: bool = False
) -> pd.DataFrame:
    if isinstance(input_dir, str):
        input_dir = Path(input_dir)

    metrics_files = list(input_dir.rglob("metric_results.csv"))
    df = pd.concat(
        (
            pd.read_csv(file, index_col=None, header=0, dtype={"patient_id": str})
            for file in metrics_files
        ),
        axis=0,
        ignore_index=True,
    )

    print(
        f"collected metric results of {len(df.study_inst_uid.unique())} studies ({len(df.series_inst_uid.unique())} series)"
    )

    if write_to_csv:
        filepath = Path(input_dir, "all_metric_results.csv")
        df.to_csv(filepath, sep=",", na_rep=nan, index=None, columns=df.columns)
        print(f"written results to `{filepath}`")

    return df
