import sys
import argparse

from src import segmentation


def get_args():
    parser = argparse.ArgumentParser(
        prog="segment",
        description="Segmentation of spine vertebrae. Segmetnation of muscle, fat tissues at L3 vertebra in axial view.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        help="path to NifTi data to segment",
        default="./inputs",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        help="path to output directory",
        default="./outputs",
    )
    parser.add_argument(
        "--slices-num",
        type=int,
        help=(
            "total number of slices to extract along superior/inferior direction with centroid_z_index at middle slice\n"
            "eg: slices_num=5, centroid_z_index=30 extracts slices in range (28, 32)\n"
            "for usage must be >= 2"
        ),
        default=0,
    )
    # parser.add_argument(
    #     "-am",
    #     "--additional-metrics",
    #     nargs="+",
    #     help="space separated list of additional metrics to compute, see docs or run segment.py --help-metrics for description",
    #     metavar="metrics",
    #     choices=("smi"),
    # )
    # parser.add_argument(
    #     "--save-segmentations",
    #     action="store_true",
    #     help="save segmentation masks",
    #     default=False,
    # )
    parser.add_argument(
        "--save-mask-overlays",
        action="store_true",
        help="save overlayed segmentation masks",
        default=False,
    )
    parser.add_argument(
        "--collect-metric-results",
        action="store_true",
        help="collect all metric_results.csv into one .csv file",
        default=False,
    )
    return parser.parse_args()


def check_slices_num_value(arg):
    value = int(arg)
    if value != 0 and value < 2:
        raise ValueError(f"argument {arg=} must be greater or equal 0")


if __name__ == "__main__":
    args = get_args()

    check_slices_num_value(args.slices_num)

    segmentation.segment_ct_study(
        args.input_dir,
        args.output_dir,
        slices_num=args.slices_num,
        save_mask_overlays=args.save_mask_overlays,
        collect_metric_results=args.collect_metric_results,
    )
    print("finished CT segmentation")
