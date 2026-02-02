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
        required=True,
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
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    verbose = args.verbose
    main_logger = slogger.get_logger(__name__)

    participant_list = utils.read_patient_list(args.participant_list)
    if not participant_list:
        sys.exit(-1)

    labkey_api = database.labkey_from_dotenv(verbose=verbose)
    if not labkey_api.is_labkey_reachable():
        main_logger.critical("labkey is unreachable")
        sys.exit(-1)

    queried_labkey_response: list[database.LabkeyRow] = labkey_api._select_rows(
        schema_name="lists",
        query_name="RDG-CT-Sarko-All",
        columns=[
            "STUDY_INSTANCE_UID",
            "VYSKA_PAC.",
            "PARTICIPANT",
            "PACS_CISLO",
        ],  # [TODO]: possibly add CAS_VYSETRENI
        filter_dict={"PARTICIPANT": participant_list},
        sanitize_rows=True,
    )

    if queried_labkey_response is None:
        main_logger.critical(
            "quitting sarcoseg, labkey query response has no StudyInstanceUIDs"
        )
        sys.exit(-1)

    pacs_api = pacs.pacs_from_dotenv(verbose=verbose)
    output_dir = Path(args.output_dir, timestamp)
    output_dir.mkdir(exist_ok=True)

    for labkey_data in queried_labkey_response:
        input_study_dir = Path(args.input_dir, labkey_data.study_instance_uid)

        if not input_study_dir.exists() and list(input_study_dir.rglob("*")) != 0:
            main_logger.info(
                f"input study directory `{input_study_dir}` not found, trying to download from PACS instead"
            )

            status = pacs_api._movescu(
                labkey_data.study_instance_uid,
                input_study_dir,
            )

            if status == -1:
                continue

        output_study_dir = Path(output_dir, labkey_data.study_instance_uid)

        main_logger.info(
            f"preprocessing study {labkey_data.study_instance_uid} patient {labkey_data.participant}"
        )
        study_data = preprocessing.preprocess_dicom_study(
            input_study_dir,
            output_study_dir,
            labkey_data,
        )

        main_logger.info(
            f"segmenting study {labkey_data.study_instance_uid} for patient {labkey_data.participant}"
        )
        metrics_results = segmentation.segment_ct_study(
            output_study_dir, output_study_dir, save_mask_overlays=True
        )

        if args.upload_labkey:
            # [TODO]: send dicom data to labkey
            # [TESTME]:
            labkey_api._upload_data(
                schema_name="lists",
                query_name="CTVysetreni",
                rows=study_data._to_list_of_dicts(),
            )

            # [TODO]: send segmentation data to labkey
            # [TESTME]:
            metrics_data = [d._to_dict() for d in metrics_results]
            labkey_api._upload_data(
                schema_name="lists",
                query_name="CTSegmentationData",
                rows=metrics_data,
            )

        if args.remove_dicom_files:
            utils.remove_dicom_dir(input_study_dir)

    if not any(output_dir.iterdir()):
        utils.remove_empty_segmentation_dir(output_dir)


if __name__ == "__main__":
    args = get_args()
    main(args)
