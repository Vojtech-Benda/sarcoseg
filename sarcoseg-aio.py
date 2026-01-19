import sys
import argparse
from pathlib import Path
from datetime import datetime

from src import preprocessing
from src import segmentation
from src import utils
from src.network import pacs, database
from src import slogger


def get_args():
    parser = argparse.ArgumentParser(
        prog="sarcoseg",
        description="segmentation of L3 axial tissues",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="print more information to console",
        default=False,
    )
    # parser.add_argument(
    #     "-dd",
    #     "--download-dir",
    #     type=str,
    #     help="path to PACS download directory",
    #     default="./download",
    # )
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
        "-pl",
        "--patient-list",
        type=str,
        help="path to list of labkey data (.csv/.txt)",
        required=False,
    )
    parser.add_argument(
        "--upload-labkey",
        action="store_true",
        help="upload preprocessing and segmentation data to labkey",
        default=False,
    )
    parser.add_argument(
        "--remove-dicom-files",
        action="store_true",
        help="remove DICOM files from input directory",
        default=False,
    )

    return parser.parse_args()


def main(args: argparse.Namespace):
    verbose = args.verbose
    main_logger = slogger.get_logger(__name__)

    patient_id_list = utils.read_patient_list(args.patient_list)
    if not patient_id_list:
        sys.exit(-1)

    labkey_api = database.labkey_from_dotenv()
    if not labkey_api.is_labkey_reachable():
        main_logger.critical("labkey is unreachable")
        sys.exit(-1)

    queried_labkey_data: list[database.LabkeyRow] = labkey_api._select_rows(
        schema_name="lists",
        query_name="RDG-CT-Sarko-All",
        columns=[
            "RODNE_CISLO",
            "CAS_VYSETRENI",
            "STUDY_INSTANCE_UID",
            "VYSKA_PAC.",
            "PARTICIPANT",
            "PACS_CISLO",
        ],
        filter_dict={"RODNE_CISLO": patient_id_list},
        sanitize_rows=True,
    )

    if queried_labkey_response is None:
        main_logger.critical(
            "quitting sarcoseg, labkey query response has no StudyInstanceUIDs"
        )
        sys.exit(-1)

    pacs_api = pacs.pacs_from_dotenv(verbose=verbose)
    output_dir = Path(args.output_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    output_dir.mkdir(exist_ok=True)

    for labkey_data in queried_labkey_data:
        input_study_dir = Path(args.input_dir, labkey_data.study_instance_uid)

        if not input_study_dir.exists():
            main_logger.info(
                f"input study directory `{input_study_dir}` not found, trying to download from PACS instead"
            )

            status = pacs_api._movescu(
                labkey_data.study_instance_uid,
                str(input_study_dir),
            )

            if status == -1:
                continue

        output_study_dir = Path(output_dir, labkey_data.study_instance_uid)

        main_logger.info(
            f"preprocessing study {labkey_data.study_instance_uid} patient {labkey_data.patient_id}"
        )
        dicom_study_tags = preprocessing.preprocess_dicom_study(
            input_study_dir,
            output_study_dir,
            labkey_data,
        )

        main_logger.info(
            f"segmenting study {labkey_data.study_instance_uid} for patient {labkey_data.patient_id}"
        )
        all_metrics_results = segmentation.segment_ct_study(
            output_study_dir, output_study_dir, save_mask_overlays=True
        )

        """
        if args.upload_labkey:
            pass
            # TODO: send dicom data to labkey
            # TESTME:
            labkey_api._upload_data(
                schema_name="lists", query_name="CTVysetreni", rows=dicom_study_tags
            )

            # TODO: send segmentation data to labkey
            # TESTME:
           
            # labkey_api._upload_data(
            #     schema_name="lists", query_name="CTSegmentationData", rows=segment_data
            # )
           
        """

        if args.remove_dicom_files:
            utils.remove_dicom_dir(input_study_dir)

    if not any(output_dir.iterdir()):
        utils.remove_empty_segmentation_dir(output_dir)


if __name__ == "__main__":
    args = get_args()
    main(args)
