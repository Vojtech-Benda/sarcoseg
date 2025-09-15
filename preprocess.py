import sys
import argparse

from src import preprocessing


def get_args():
    parser = argparse.ArgumentParser(
        prog="preprocess",
        description="preprocess DICOM files and save valid CT series as NifTi",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input-dir", type=str, help="path to DICOM files", required=True
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        help="path to save preprocessed NifTi files",
        default="./inputs",
    )
    parser.add_argument(
        "--collect-dicom-tags",
        action="store_true",
        help="collect all DICOM tags into one table file at <output-dir>/<dicom_tags_d-m-Y_H-M-S.csv>",
        default=False,
    )
    parser.add_argument(
        "-ql",
        "--query-labkey",
        action="store_true",
        help="query labkey for patient data, see docs for queried tables and data",
        default=False,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    preprocessing.preprocess_dicom(
        args.input_dir, args.output_dir, query_labkey=args.query_labkey
    )

    if args.collect_dicom_tags:
        preprocessing.collect_all_dicom_tags(args.output_dir)
