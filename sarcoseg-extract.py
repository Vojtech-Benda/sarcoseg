import logging
import shutil
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Iterable

from src.slogger import setup_logger

FILETYPES = {
    "nifti": (
        "input_ct_volume",
        "spine_mask",
        "tissue_mask",
        "tissue_mask_pp",
        "tissue_slices",
    ),
    "images": (
        "tissue_slice",
        "tissue_overlay",
        "spine_coronal_overlay",
        "spine_sagittal_overlay",
    ),
    "dicom-tags": ("dicom_tags",),
    "metrics": ("metrics",),
    "report": ("report",),
}

FILE_EXTS = {
    "nifti": "nii.gz",
    "images": "png",
    "dicom-tags": "json",
    "metrics": "json",
    "report": "csv",
}


def get_args():
    parser = ArgumentParser(
        "sarcoseg-extract",
        description="Extract sarcoseg-aio JSON files, NifTI files or images. Files are stored in results directory under extract_<parent-dir>",
    )
    parser.add_argument(
        "-r", "--results-dir", help="path to directory with results", required=True
    )
    parser.add_argument(
        "-f",
        "--files",
        nargs="*",
        choices=[
            item
            for v in FILETYPES.values()
            for item in (v if isinstance(v, tuple) else [v])
        ],
        help="space separated list of file types to extract",
        default=[],
    )
    parser.add_argument(
        "-t", "--types", nargs="*", choices=FILETYPES.keys(), default=[]
    )
    parser.add_argument(
        "-ov", "--overwrite", action="store_true", help="overwrite extracted results"
    )

    return parser.parse_args()


def copy_files(
    input_dir: Path, output_dir: Path, file_type: str, specific_files: Iterable
):
    src_paths = [
        p
        for basename in specific_files
        for p in input_dir.rglob(basename + "*." + FILE_EXTS[file_type])
        if p.is_file()
    ]
    dst_paths = [output_dir.joinpath(Path(*p.parts[1:])) for p in src_paths]

    # TODO: replace assert with "if len(src_image_paths) == len(dst_image_paths)" block check? return exception/None/something else on failure
    assert len(src_paths) == len(dst_paths)

    for src, dst in zip(src_paths, dst_paths):
        dst.parent.mkdir(exist_ok=True, parents=True)

        shutil.copy2(src, dst)


def main(args: Namespace):
    setup_logger()
    log = logging.getLogger("sarcoseg-extract")

    overwrite = args.overwrite
    if overwrite:
        log.warning("overwriting enabled")

    results_dirpath = Path(args.results_dir)

    if not results_dirpath.exists() or not results_dirpath.is_dir():
        log.error(
            f"results directory `{results_dirpath}` not found or is not directory"
        )
        sys.exit(-1)

    files = args.files
    filetypes = args.types
    if not (files or filetypes):
        log.error("arguments --files (-f) or --types (-t) not used")
        sys.exit(-1)

    log.info(f"extracting these files: {files}")
    log.info(f"extracting these file types: {filetypes}")

    extract_dirpath = results_dirpath.joinpath("extract_results")
    extract_dirpath.mkdir(exist_ok=args.overwrite)
    log.info(f"copying results into `{extract_dirpath}`")

    for file_type in filetypes:
        specific_files = [file for file in files if file in FILETYPES[file_type]]
        if not specific_files:
            specific_files = FILETYPES[file_type]

        copy_files(results_dirpath, extract_dirpath, file_type, specific_files)


if __name__ == "__main__":
    args = get_args()
    main(args)
