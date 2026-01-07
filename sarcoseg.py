import sys
import argparse
import warnings
from pathlib import Path
from datetime import datetime

from src import preprocessing
from src import segmentation
from src.network import pacs, database


def get_args():
    parser = argparse.ArgumentParser(
        prog="sarcoseg",
        description="segmentation of L3 axial tissues",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="print more information to console",
        default=False,
    )
    parser.add_argument(
        "-dd",
        "--download-dir",
        type=str,
        help="path to PACS download directory",
        default="./download",
    )
    parser.add_argument(
        "-id",
        "--input-dir",
        type=str,
        help="path to input directory",
        default="./inputs",
    )
    parser.add_argument(
        "-od",
        "--output-dir",
        type=str,
        help="path to segmentation output directory",
        default="./outputs",
    )
    parser.add_argument(
        "--upload-labkey",
        action="store_true",
        help="upload preprocessing and segmentation data to labkey",
        default=False,
    )

    return parser.parse_args()


def main(args: argparse.Namespace):
    verbose = args.verbose

    pacs_api = pacs.pacs_from_dotenv(verbose=verbose)

    labkey_api = database.labkey_from_dotenv()
    if not labkey_api.is_labkey_reachable():
        warnings.warn("Labkey is unreachable")
        sys.exit(-1)

    queried_labkey_data: list[dict] = labkey_api._select_rows(
        args.schema_name,
        args.query_name,
        columns="STUDY_INSTANCE_UID,VYSKA_PAC.,PARTICIPANT",
        sanitize_rows=True,
    )

    if queried_labkey_data is None:
        warnings.warn("exiting sarcoseg")
        warnings.warn(
            "reason: no labkey response data with queried study instance uids"
        )
        sys.exit(-1)

    segmentation_output_dir = Path(
        args.output_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    segmentation_output_dir.mkdir(exist_ok=True)

    for labkey_data in queried_labkey_data:
        study_uid = labkey_data["STUDY_INSTANCE_UID"]
        download_dir = Path(args.download_dir, study_uid)
        status = pacs_api.movescu(
            study_uid,
            download_dir,
        )

        if status == -1:
            continue

        preprocessing.preprocess_dicom_study(download_dir, args.input_dir, labkey_data)
        
        preprocessing.collect_all_dicom_tags("")

        segmentation.segment_ct_study(
            args.input_dir, segmentation_output_dir, save_mask_overlays=True
        )

    if args.upload_labkey:
        pass
        # TODO: send dicom data to labkey
        # TODO: send segmentation data to labkey


if __name__ == "__main__":
    args = get_args()
    main(args)
