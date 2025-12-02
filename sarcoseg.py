import sys
import argparse
import warnings
from pathlib import Path

from src import preprocessing
from src import segmentation
from src.network import pacs
from src import database


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
        default="./inputs",
    )
    parser.add_argument(
        "-od",
        "--output-dir",
        type=str,
        help="path to segmentation output directory",
        default="./outputs",
    )
    return parser.parse_args()


def main(args):
    verbose = args.verbose

    pacs_api = pacs.pacs_from_dotenv(verbose=verbose)

    labkey_api = database.labkey_from_dotenv()
    if not labkey_api.is_labkey_reachable():
        warnings.warn("Labkey is unreachable")
        sys.exit(-1)

    patient_data: list[database.LabkeyRow] = labkey_api.query_patients(
        args.schema_name,
        args.query_name,
        columns="RODNE_CISLO,CAS_VYSETRENI,STUDY_INSTANCE_UID,VAHA_PAC.,PARTICIPANT",
    )

    if len(patient_data) == 0:
        warnings.warn("exiting sarcoseg")
        warnings.warn("reason: no labkey response data")
        sys.exit(-1)

    for patient in patient_data:
        download_dir = Path(args.download_directory, patient.study_instance_uid)
        status = pacs_api.movescu(
            patient.study_instance_uid,
            patient.patient_id,
            patient.study_date,
            download_dir,
        )

        if status == -1:
            continue

        preprocessing.preprocess_dicom(
            download_dir,
            download_dir,
            patient,
        )

        segmentation.segment_ct(download_dir, args.output_dir)


def sarcoseg():
    pass


if __name__ == "__main__":
    args = get_args()
    main(args)
