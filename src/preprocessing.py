import sys
import re

from pathlib import Path
from typing import Any, Union
from dataclasses import dataclass

import pandas as pd
import pydicom
from statistics import mean
from dicom2nifti.convert_dicom import dicom_array_to_nifti
from datetime import datetime


SERIES_DESC_PATTERN = re.compile(
    r"|".join(
        ("protocol", "topogram", "scout", "patient", "dose", "report", "monitor")
    ),
    re.IGNORECASE,
)
CONTRAST_PHASES_PATTERN = re.compile(
    r"|".join(("abdomen", "arterial", "nephro", "venous")), re.IGNORECASE
)


@dataclass
class SeriesData:
    series_inst_uid: str = None
    series_description: str = None
    filepaths: list[Path] = None
    num_of_filepaths: int = None
    slice_thickness: float = None
    has_contrast: str = None
    contrast_phase: str = None
    kilo_voltage_peak: float = None
    mean_tube_current: float = None
    irradiation_event_uid: str = None
    mean_ctdi_vol: float = None
    dose_length_product: float = None


@dataclass
class StudyData:
    patient_id: str = None
    study_inst_uid: str = None
    study_date: str = None
    series_dict: dict[str, SeriesData] = None


def preprocess_dicom(
    input_dir: Union[str, Path], output_dir: Union[str, Path] = "./inputs"
):
    print("preprocessing DICOM files")

    if not isinstance(input_dir, Path):
        input_dir = Path(input_dir)

    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)

    for dicom_dir in input_dir.glob("*/"):
        dicom_files = find_dicoms(dicom_dir)

        study_data, files_by_series_uid, dose_report_path = filter_dicom_files(
            dicom_files
        )

        dose_per_event = extract_dose_values(dose_report_path)

        study_data.series_dict = select_series_to_segment(
            files_by_series_uid, dose_report=dose_per_event
        )

        print(f"found {len(study_data.series_dict)} valid series for segmentation")
        print(
            f"saving id `{study_data.patient_id}`, study instance uid `{study_data.study_inst_uid}`"
        )

        output_study_dir = Path(output_dir, study_data.study_inst_uid)
        output_study_dir.mkdir(exist_ok=True, parents=True)

        write_dicom_tags(study_data, output_study_dir)

        for series_data in study_data.series_dict.values():
            print(
                f"series instance uid `{series_data.series_inst_uid}`\n"
                f"series description `{series_data.series_description}`\n"
                f"contrast phase `{series_data.has_contrast}`, type `{series_data.contrast_phase}`"
            )

            dicom_datasets = [pydicom.dcmread(file) for file in series_data.filepaths]

            output_series_dir = output_study_dir.joinpath(series_data.series_inst_uid)
            output_series_dir.mkdir(exist_ok=True, parents=True)
            output_filepath = output_series_dir.joinpath("input_volume.nii.gz")

            if output_filepath.exists():
                print(
                    f"overwriting existing input_volume.nii.gz at {str(output_filepath)}"
                )

            try:
                dicom_array_to_nifti(
                    dicom_list=dicom_datasets,
                    output_file=output_filepath,
                    reorient_nifti=True,
                )
                print(f"nifti written to `{output_filepath}`\n")
            except RuntimeError as err:
                print(err)

        print("-" * 25)


def filter_dicom_files(
    dicom_files: list[Path],
) -> tuple[StudyData, dict[str, list[Path]], Path]:
    """
    Sorts filepaths by DICOM tag SeriesInstanceUID.
    Also returns fullpath of Dose report series if found.

    Args:
        dicom_files (list[Path]): List of dicom files, excluding DICOMDIR file.

    Returns:
        tuple:
            study_data StudyData: dataclass containing basic study metadata.\n
            files_by_uid dict[str, list[Path]]: Dictionary mapping each SeriesInstanceUID to a list of filepaths.\n
            dose_report_file Path: Path to Dose Report series file, if found, otherwise None.
    """

    files_by_uid: dict[str, list[str]] = {}

    ds = pydicom.dcmread(
        dicom_files[0],
        stop_before_pixels=True,
        specific_tags=["PatientID", "StudyDate", "StudyInstanceUID"],
    )

    study_data = StudyData(
        patient_id=ds.PatientID,
        study_inst_uid=ds.StudyInstanceUID,
        study_date=ds.StudyDate,
    )

    dose_report_file = None
    for file in dicom_files:
        ds = pydicom.dcmread(
            file,
            stop_before_pixels=True,
            specific_tags=[
                "ImageType",
                "SeriesInstanceUID",
                "SeriesDescription",
                "Modality",
                "SliceThickness",
            ],
        )
        series_uid = ds.SeriesInstanceUID

        # filter out files in series matching pattern:
        # ("protocol", "topogram", "scout", "patient", "dose", "report"), ignoring case
        if "dose report" in ds.SeriesDescription.lower():
            dose_report_file = file
            continue

        if SERIES_DESC_PATTERN.search(ds.SeriesDescription):
            continue

        if not hasattr(ds, "SliceThickness"):
            continue

        if "DERIVED" in ds.ImageType:
            continue

        if series_uid in files_by_uid:
            files_by_uid[series_uid].append(file)
        else:
            files_by_uid[series_uid] = [file]

    print(f"\nid '{study_data.patient_id}' filtered {len(files_by_uid.keys())} series")
    return study_data, files_by_uid, dose_report_file


def select_series_to_segment(
    all_series: dict,
    dose_report: dict,
) -> dict[str, SeriesData]:
    """
    Select one or multiple series for segmentation and extract their DICOM tags. Finds the native CT series with the lowest slice thickness and highest number of slices.
    Also finds the contrast phase CT series with the lowest slice thickness and highest number of slices per each contrast phase found.

    Args:
        dataset (pydicom.FileDataset): all series found.

    Returns:
        list: list consisting of SeriesMetadata dataclass for native CT and each contract phase CT.
    """

    series_by_contrast: dict[str, list[SeriesData]] = {}

    for series_uid, filepaths in all_series.items():
        # read only the first file to filter series
        dataset = pydicom.dcmread(filepaths[0], stop_before_pixels=True)

        # filter by words in SeriesDescription
        series_desc: str = dataset.SeriesDescription

        contrast_applied = dataset.get("ContrastBolusAgent", None)

        series_data = SeriesData(
            series_inst_uid=series_uid,
            series_description=series_desc,
            slice_thickness=float(dataset.SliceThickness),
            filepaths=filepaths,
            num_of_filepaths=len(filepaths),
            has_contrast="yes" if contrast_applied else "no",
            irradiation_event_uid=dataset.get("IrradiationEventUID", "none"),
        )

        contrast_match = CONTRAST_PHASES_PATTERN.search(series_desc)
        if contrast_match:
            series_data.contrast_phase = contrast_match.group().lower()
        else:
            series_data.contrast_phase = "other"

        if series_data.contrast_phase in series_by_contrast:
            series_by_contrast[series_data.contrast_phase].append(series_data)
        else:
            series_by_contrast[series_data.contrast_phase] = [series_data]

    selected_series = {
        phase: min(
            data_list,
            key=lambda data: (data.slice_thickness, -data.num_of_filepaths),
        )
        for phase, data_list in series_by_contrast.items()
    }

    for series_data in selected_series.values():
        tube_currents = [
            pydicom.dcmread(
                p, stop_before_pixels=True, specific_tags=["XRayTubeCurrent"]
            ).get("XRayTubeCurrent", None)
            for p in series_data.filepaths
        ]
        series_data.mean_tube_current = mean(
            [float(current) for current in tube_currents if current]
        )

        series_data.kilo_voltage_peak = float(
            pydicom.dcmread(
                series_data.filepaths[0], stop_before_pixels=True, specific_tags=["KVP"]
            ).get("KVP", 0.0)
        )

        series_data.mean_ctdi_vol = dose_report[series_data.irradiation_event_uid][
            "mean_ctdi_vol"
        ]
        series_data.dose_length_product = dose_report[
            series_data.irradiation_event_uid
        ]["dlp"]

    return selected_series


def extract_dose_values(dose_report: str) -> dict[str, float]:
    """
    Map IrradiationEventUID to dose values

    """
    ds = pydicom.dcmread(dose_report)
    event_to_dose = {}

    if ds.Modality != "SR":
        raise ValueError("file is not dose report")

    def walk_content(seq, current_event=None):
        for item in seq:
            vr = item.ValueType

            # get irradiation event uid
            if hasattr(item, "ConceptNameCodeSequence"):
                code_meaning = item.ConceptNameCodeSequence[0].CodeMeaning
                if code_meaning == "Irradiation Event UID" and hasattr(item, "UID"):
                    current_event = item.UID
                    if current_event not in event_to_dose:
                        event_to_dose[current_event] = {
                            "dlp": None,
                            "mean_ctdi_vol": None,
                        }

            # find dose values
            if vr == "NUM" and hasattr(item, "MeasuredValueSequence"):
                code_meaning = item.ConceptNameCodeSequence[0].CodeMeaning
                val = float(item.MeasuredValueSequence[0].NumericValue)

                if current_event:
                    if code_meaning in ["DLP"]:
                        event_to_dose[current_event]["dlp"] = val
                    elif code_meaning in ["Mean CTDIvol"]:
                        event_to_dose[current_event]["mean_ctdi_vol"] = val

            # recursively iterate content sequences
            if hasattr(item, "ContentSequence"):
                walk_content(item.ContentSequence, current_event)

    walk_content(ds.ContentSequence)
    return event_to_dose


def find_dicoms(dicom_dir: Path):
    for root, _, files in dicom_dir.walk():
        if len(files) == 0 or "DICOMDIR" in files:
            continue
        return list(root.iterdir())


def write_dicom_tags(study: StudyData, study_dir: Path):
    rows: list[dict[str, Any]] = []
    for _, series in study.series_dict.items():
        row = {
            "patient_id": study.patient_id,
            "study_inst_uid": study.study_inst_uid,
            "study_date": study.study_date,
            "series_inst_uid": series.series_inst_uid,
            "series_description": series.series_description,
            "slice_thickness": series.slice_thickness,
            "has_contrast": series.has_contrast,
            "contrast_phase": series.contrast_phase,
            "kilo_voltage_peak": series.kilo_voltage_peak,
            "mean_tube_current": series.mean_tube_current,
            "mean_ctdi_vol": series.mean_ctdi_vol,
            "dose_length_product": series.dose_length_product,
        }
        rows.append(row)

    df = pd.DataFrame(rows, columns=rows[0].keys())
    filepath = study_dir.joinpath("dicom_tags.csv")

    if filepath.exists():
        print(f"overwriting existing dicom_tags.csv at `{str(filepath)}`")

    df.to_csv(
        filepath,
        sep=",",
        na_rep="nan",
        index=False,
        columns=df.columns,
    )
    print(
        f"id `{study.patient_id}` study instance uid `{study.study_inst_uid}` ({len(df.series_inst_uid.unique())} series) written DICOM tags to {filepath}"
    )


def collect_all_study_tags(input_dir: Union[str, Path]):
    if not isinstance(input_dir, Path):
        input_dir = Path(input_dir)

    dicom_tags_files = list(input_dir.rglob("dicom_tags.*"))
    df = pd.concat(
        (pd.read_csv(file, index_col=None, header=0) for file in dicom_tags_files),
        axis=0,
        ignore_index=True,
    )

    filepath = Path(
        "./outputs",
        "all_dicom_tags" + datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + ".csv",
    )
    df.to_csv(filepath, sep=",", na_rep="nan", index=None, columns=df.columns)
    print(
        f"written all DICOM tags of {len(df.study_inst_uid.unique())} studies ({len(df.series_inst_uid.unique())} series) to `{filepath}`"
    )


if __name__ == "__main__":
    if len(sys.argv) < 1:
        print("Missing input directory")
        sys.exit(-1)
    preprocess_dicom(sys.argv[1])
