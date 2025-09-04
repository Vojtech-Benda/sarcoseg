from typing import Union, Any
from pathlib import Path
from time import perf_counter
from datetime import datetime
import shutil
import pandas as pd
from totalsegmentator.python_api import totalsegmentator

import nibabel as nib
from numpy import nan
from nnunet.inference.predict import predict_cases

from src import segmentation
from src import visualization
from src import utils
from src.utils import DEFAULT_VERTEBRA_CLASSES

MODEL_DIR = Path("models", "muscle_fat_tissue_stanford_0_0_2")


def segment_ct(
    input_dir: str,
    output_dir: str,
    additional_metrics: list,
    slices_num: int = 0,
    save_segmentations: bool = False,
    save_mask_overlays: bool = False,
):
    case_dirs = list(Path(input_dir).glob("*/"))
    print(f"found {len(case_dirs)} case directories")

    for case_dir in case_dirs:
        ct_volume_paths = list(case_dir.rglob("*/*.nii.gz"))
        print("\n" + "-" * 25)
        print(
            f"\nfound {len(ct_volume_paths)} volumes to segment spine for case {case_dir.name}"
        )

        metric_results_list: list[dict[str, Any]] = []

        for ct_volume_path in ct_volume_paths:
            # construct output path from [input_dir, study_inst_uid, series_inst_uid, file]
            case_output_dir = Path(output_dir, *ct_volume_path.parts[1:-1])
            case_images_dir = case_output_dir.joinpath("images")
            case_output_dir.mkdir(exist_ok=True, parents=True)
            case_images_dir.mkdir(exist_ok=True)

            shutil.copy2(ct_volume_path, case_output_dir.joinpath(ct_volume_path.name))

            spine_results = segmentation.segment_spine(
                ct_volume_path,
                case_output_dir,
            )

            spine_mask = (
                spine_results["spine_mask"]
                if "spine_mask" in spine_results
                else spine_results["spine_mask_path"]
            )

            slice_results = utils.extract_slices(
                ct_volume_path, case_output_dir, spine_mask, slices_num
            )

            tissue_results = segmentation.segment_tissues(
                slice_results["sliced_volume_path"], case_output_dir
            )

            postproc_results = utils.postprocess_tissue_masks(
                tissue_results["mask"],
                tissue_results["volume"],
                tissue_results["mask_filepath"],
            )

            metric_results = utils.compute_metrics(
                postproc_results["processed_mask"],
                tissue_results["volume"],
                metrics=additional_metrics,
                spacing=tissue_results["spacing"],
            )

            duration = (
                spine_results["duration"]
                + slice_results["duration"]
                + tissue_results["duration"]
                + postproc_results["duration"]
            )

            study_inst_uid, series_inst_uid = ct_volume_path.parts[1:-1]
            metric_results["duration"] = duration
            metric_results["study_inst_uid"] = study_inst_uid
            metric_results["series_inst_uid"] = series_inst_uid
            metric_results_list.append(metric_results)

            if save_mask_overlays:
                visualization.overlay_spine_mask(
                    ct_volume_path,
                    spine_results["spine_mask_path"],
                    slice_results["vert_centroid"],
                    output_dir=case_images_dir,
                )

                visualization.overlay_tissue_mask(
                    slice_results["sliced_volume_path"],
                    tissue_results["mask_filepath"],
                    output_dir=case_images_dir,
                )

        write_metric_results(
            metric_results_list, Path(output_dir, ct_volume_path.parts[1])
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

    spine_mask_path = output_dir.joinpath("spine_mask.nii.gz")

    if spine_mask_path.exists() and not overwrite_output:
        print(f"file '{spine_mask_path}' exists, skipping spine segmentation")
        return {"spine_mask_path": spine_mask_path, "duration": 0.0}

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
    tissue_volume_path: Union[Path, str], case_output_dir: Union[Path, str]
):
    if not isinstance(case_output_dir, Path):
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


def write_metric_results(metric_results: list[dict[str, Any]], output_study_dir: Path):
    rows: list[dict[str, Any]] = []
    for res in metric_results:
        rows.append(
            {
                "study_inst_uid": res["study_inst_uid"],
                "series_inst_uid": res["series_inst_uid"],
                "muscle_area": res["area"]["muscle"],
                "muscle_mean_hu": res["mean_hu"]["muscle"],
                "sat_area": res["area"]["sat"],
                "sat_mean_hu": res["mean_hu"]["sat"],
                "vat_area": res["area"]["vat"],
                "vat_mean_hu": res["mean_hu"]["vat"],
                "imat_area": res["area"]["imat"],
                "imat_mean_hu": res["mean_hu"]["imat"],
                "duration": res["duration"],
            }
        )

    df = pd.DataFrame(rows, columns=rows[0].keys())
    filepath = output_study_dir.joinpath("metric_results.csv")
    if filepath.exists():
        print(f"overwriting existing metric_results.csv at `{filepath}`")
    df.to_csv(filepath, sep=",", na_rep=nan, columns=df.columns, index=None)


def collect_all_metric_results(input_dir: Union[str, Path]):
    if not isinstance(input_dir, Path):
        input_dir = Path(input_dir)

    metrics_files = list(input_dir.rglob("metric_results.csv"))
    df = pd.concat(
        (pd.read_csv(file, index_col=None, header=0) for file in metrics_files),
        axis=0,
        ignore_index=True,
    )

    filepath = Path(
        "./outputs",
        "all_metric_results_" + datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + ".csv",
    )
    df.to_csv(filepath, sep=",", na_rep=nan, index=None, columns=df.columns)
    print(
        f"written all metric results of {len(df.study_inst_uid.unique())} studies ({len(df.series_inst_uid.unique())} series) to `{filepath}`"
    )
