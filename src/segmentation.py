from typing import Union
from pathlib import Path
from time import perf_counter

from totalsegmentator.python_api import totalsegmentator

import nibabel as nib
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

            slice_results = utils.extract_slices(
                ct_volume_path, case_nifti_dir, spine_mask, slices_num
            )

            tissue_results = segmentation.segment_tissues(
                slice_results["sliced_volume_path"], case_nifti_dir
            )

            postproc_results = utils.postprocess_tissue_masks(
                tissue_results["mask"],
                tissue_results["volume"],
                tissue_results["mask_filepath"],
            )

            metric_results = utils.compute_metrics(
                postproc_results["processed_mask"],
                metrics=additional_metrics,
                spacing=tissue_results["spacing"],
            )

            phase = str(ct_volume_path.name).removesuffix(".nii.gz")
            duration = (
                spine_results["duration"]
                + slice_results["duration"]
                + tissue_results["duration"]
                + postproc_results["duration"]
            )
            utils.collect_results(case_dir.name, phase, metric_results, duration)

            if save_mask_overlays:
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
