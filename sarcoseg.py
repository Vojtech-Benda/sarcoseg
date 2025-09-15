import sys
import argparse

from src import preprocessing
from src import segmentation
from src.setup_project import setup_project


def get_args():
    parser = argparse.ArgumentParser(
        prog="sarcoseg",
        description="segmentation of L3 axial tissues",
    )

    sub_parsers = parser.add_subparsers(dest="command", help="select command to run")

    preprocess_parser = sub_parsers.add_parser(
        "preprocess",
        help="preprocess DICOM files and save as NifTi format",
        description="preprocessing options",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    preprocess_parser.add_argument(
        "-i", "--input-dir", type=str, help="path to DICOM files", required=True
    )
    preprocess_parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        help="path to save preprocessed NifTi files",
        default="./inputs",
    )
    preprocess_parser.add_argument(
        "--collect-dicom-tags",
        action="store_true",
        help="collect all DICOM tags into one table file at <output-dir>/<dicom_tags_d-m-Y_H-M-S.csv>",
        default=False,
    )
    preprocess_parser.add_argument(
        "-ql",
        "--query-labkey",
        action="store_true",
        help="query labkey for patient data, see docs about queried tables and data",
        default=False,
    )

    segment_parser = sub_parsers.add_parser(
        "segment",
        help="segment muscle and fat tissue at L3 level in axial view",
        description="segmentation options",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    segment_parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        help="path to NifTi data to segment",
        default="./inputs",
    )
    segment_parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        help="path to output directory",
        default="./outputs",
    )
    segment_parser.add_argument(
        "--slices-num",
        type=int,
        help=(
            "total number of slices to extract along superior/inferior direction with centroid_z_index at middle slice\n"
            "eg: slices_num=5, centroid_z_index=30 extracts slices in range (28, 32)\n"
            "for usage must be >= 2"
        ),
        default=0,
    )
    segment_parser.add_argument(
        "--add-metrics",
        nargs="+",
        help="space separated list of additional metrics to compute",
        metavar="metrics",
    )
    segment_parser.add_argument(
        "--save-segmentations",
        action="store_true",
        help="save segmentation masks",
        default=False,
    )
    segment_parser.add_argument(
        "--save-mask-overlays",
        action="store_true",
        help="save overlayed segmentation masks",
        default=False,
    )
    # TODO: add/improve later
    # segment_parser.add_argument(
    #     "--copy-dicom-tags",
    #     action="store_true",
    #     help="copy <study_inst_uid> DICOM tags table file to <output_dir>/<study_inst_uid>",
    #     default=False,
    # )
    segment_parser.add_argument(
        "--collect-metric-results",
        action="store_true",
        help="collect all metric results into one table file at <output-dir>/<metric_results_d-m-Y_H-M-S.csv>",
        default=False,
    )

    setup_parser = sub_parsers.add_parser(
        "setup-project",
        help="setup the project for usage",
        description="create directories, download models, etc.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    setup_parser.add_argument(
        "-i",
        "--input-dir",
        help="path to input directory",
        type=str,
        default="./inputs",
    )
    setup_parser.add_argument(
        "-o",
        "--output-dir",
        help="path to output directory",
        type=str,
        default="./outputs",
    )
    setup_parser.add_argument(
        "-m",
        "--model-dir",
        help="path to model directory",
        type=str,
        default="./models",
    )
    setup_parser.add_argument(
        "-n",
        "--model-name",
        help="model name",
        type=str,
        default="muscle_fat_tissue_stanford_0_0_2",
    )
    setup_parser.add_argument(
        "--remove-model-zip",
        action="store_true",
        help="remove downloaded huggingface model ZIP file",
        default=False,
    )

    # TODO: add/improve later
    # sub_parsers.add_parser(
    #     "compute-metrics",
    #     help="compute metrics",
    #     description="requires segmentation data to be present",
    # )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    if args.command == "preprocess":
        preprocessing.preprocess_dicom(
            args.input_dir, args.output_dir, query_labkey=args.query_labkey
        )
        print("finished preprocessing DICOM series")

        if args.collect_dicom_tags:
            preprocessing.collect_all_study_tags(args.output_dir)

    elif args.command == "segment":
        if args.slices_num != 0 and args.slices_num < 2:
            raise ValueError(
                f"invalid value for --slices_num (used {args.slices_num}), for usage must be greater or equal 2"
            )

        segmentation.segment_ct(
            args.input_dir,
            args.output_dir,
            args.add_metrics,
            slices_num=args.slices_num,
            save_segmentations=args.save_segmentations,
            save_mask_overlays=args.save_mask_overlays,
            collect_metric_results=args.collect_metric_results,
        )
        print("finished CT segmentation")

    elif args.command == "setup-project":
        setup_project(
            args.input_dir,
            args.output_dir,
            args.model_dir,
            args.model_name,
            args.remove_model_zip,
        )
    else:
        print(f"unknown command `{args.command}`")
        sys.exit(-1)
