import sys
import argparse

from src import preprocessing
from src import segmentation


def get_args():
    parser = argparse.ArgumentParser(prog="l3smi", description="segmentation of L3 axial tissues")
    
    sub_parsers = parser.add_subparsers(dest="command", help="select command to run")
    
    preprocess_parser = sub_parsers.add_parser("preprocess",
                                               help="preprocess DICOM files and save as NifTi format",
                                               description="preprocessing options")
    preprocess_parser.add_argument("-i", 
                                   "--input_dir",
                                   type=str, 
                                   help="path to DICOM files",
                                   required=True)
    preprocess_parser.add_argument("-o", 
                                   "--output_dir",
                                   type=str, 
                                   help="path to save preprocessed NifTi files", 
                                   default="./inputs")
    preprocess_parser.add_argument("--anonymize", 
                                   action="store_true", 
                                   help="anonymize DICOM series before saving")
    
    dicom_tags = ("PatientID", "StudyInstanceUID", "StudyDate", "SeriesDescription", "SliceThickness")
    preprocess_parser.add_argument("--dicom_tags", 
                                   nargs="+", 
                                   help=f"space separated list of additional DICOM tags to extract \n(default: {dicom_tags})")    
    
    segment_parser = sub_parsers.add_parser("segment", 
                                            help="segment muscle and fat tissue at L3 level in axial viewl",
                                            description="segmentation options")
    segment_parser.add_argument("-i",
                                "--input_dir",
                                type=str,
                                help="path to NifTi data to segment",
                                default="./inputs")
    segment_parser.add_argument("-o",
                                "--output_dir",
                                type=str, 
                                help="path to output directory",
                                default="./outputs")
    segment_parser.add_argument("--slices_num",
                                type=int, 
                                help=("number of slices in superior AND inferior direction from centroid index to extract for tissue segmentation\n"
                                      "example: slices_num=10, extract [centroid_index - 10:centroid_index + 10], segmentation over 20 slices"),
                                default=0)
    segment_parser.add_argument("--add_metrics",
                                nargs="+",
                                help="space separated list of additional metrics to compute", 
                                metavar="metrics")
    segment_parser.add_argument("--save_segmentations", 
                                action="store_true", 
                                help="save segmentation masks")

    setup_parser = sub_parsers.add_parser("setup",
                                          help="setup the project for usage"
                                          )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    
    if args.command == "preprocess":
        preprocessing.preprocess_dicom(
            args.input_dir, 
            args.output_dir, 
            anonymize=args.anonymize, 
            dicom_tags=args.dicom_tags
            )
        print("finished preprocessing DICOM series")
        
    elif args.command == "segment":
        segmentation.segment_ct(
            args.input_dir,
            args.output_dir,
            args.add_metrics,
            save_segmentation=args.save_segmentations,
            slices_num=args.slices_num
            )
        print("finished CT segmentation")
        
    else:
        print(f"unknown command '{args.command}'")
        sys.exit(-1)    