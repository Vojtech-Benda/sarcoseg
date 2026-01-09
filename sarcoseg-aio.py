import sys
import argparse
import warnings
from pathlib import Path
from datetime import datetime

from src import preprocessing
from src import segmentation
from src.network import pacs, database
from src.utils import remove_empty_segmentation_dir


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

    # labkey_api = database.labkey_from_dotenv()
    # if not labkey_api.is_labkey_reachable():
    #     warnings.warn("Labkey is unreachable")
    #     sys.exit(-1)

    # queried_labkey_data: list[dict] = labkey_api._select_rows(
    #     schema_name="lists",
    #     query_name="RDG-CT-Sarko-All",
    #     columns="STUDY_INSTANCE_UID,VYSKA_PAC.,PARTICIPANT",
    #     sanitize_rows=True,
    # )

    # if queried_labkey_data is None:
    #     warnings.warn("exiting sarcoseg")
    #     warnings.warn(
    #         "reason: no labkey response data with queried study instance uids"
    #     )
    #     sys.exit(-1)

    segmentation_output_dir = Path(
        args.output_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    segmentation_output_dir.mkdir(exist_ok=True)

    queried_labkey_data = [
        {
            "PARTICIPANT": "PAT0002",
            "STUDY_INSTANCE_UID": "1.2.276.0.7230010.3.1.2.3400784247.793340.1741033467.263",
            "VYSKA_PAC.": "180.0",
        },
    ]

    for labkey_data in queried_labkey_data:
        study_uid = labkey_data["STUDY_INSTANCE_UID"]
        download_dir = Path(args.download_dir, study_uid)
        status = pacs_api._movescu(
            study_uid,
            str(download_dir),
        )

        # if status == -1:
        #     continue

        dicom_study_tags = preprocessing.preprocess_dicom_study(
            download_dir, segmentation_output_dir, labkey_data
        )

        print(dicom_study_tags.patient_id, dicom_study_tags.study_inst_uid)
        # all_metrics_results = segmentation.segment_ct_study(
        #     args.input_dir, segmentation_output_dir, save_mask_overlays=True
        # )

        # dicom_data = preprocessing.collect_all_dicom_tags("")
        # segmentation_results = segmentation.collect_all_metric_results()

    if not any(segmentation_output_dir.iterdir()):
        remove_empty_segmentation_dir(segmentation_output_dir)

    if args.upload_labkey:
        pass
        # TODO: send dicom data to labkey
        # TESTME:
        """
        labkey_api._upload_data(
            schema_name="lists", query_name="CTVysetreni", rows=dicom_data
        )
        """
        # TODO: send segmentation data to labkey
        # TESTME:
        """
        labkey_api._upload_data(
            schema_name="lists", query_name="CTSegmentationData", rows=segment_data
        )
        """


if __name__ == "__main__":
    args = get_args()
    main(args)
