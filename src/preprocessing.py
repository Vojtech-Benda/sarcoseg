import sys
import re

from pathlib import Path
from typing import Union
from dataclasses import dataclass

import pydicom
import pandas as pd
from statistics import mean
from time import perf_counter
from dicom2nifti.common import sort_dicoms, validate_orientation
from dicom2nifti.convert_dicom import dicom_array_to_nifti


SERIES_DESC_PATTERN = re.compile(
    r"|".join(("protocol", "topogram", "scout", "patient", "dose", "report")),
    re.IGNORECASE,
)
CONTRAST_PHASES = re.compile(r"|".join(("arterial", "nephro", "venous")), re.IGNORECASE)


@dataclass
class SeriesMetadata:
    patient_id: str = None
    study_instance_uid: str = None
    study_date: str = None
    series_instance_uid: str = None
    series_description: str = None
    filepaths: list[Union[str, Path]] = None
    num_of_files: int = None
    slice_thickness: Union[float, None] = None
    has_contrast: bool = False
    contrast_phase: str = None
    kilo_voltage_peak: Union[float, None] = None
    irradiation_event_uid: str = None
    mean_ctdi_vol: Union[float, None] = None
    dose_length_product: Union[float, None] = None

    additional_tags = {}

    def print_data(self, print_additional_tags=False):
        msg = (
            f"patient id: {self.patient_id}\n"
            f"series description: {self.series_description}\n"
            f"number of files: {self.num_of_files}\n"
            f"slice thickness: {self.slice_thickness}\n"
            f"has contrast: {self.has_contrast}\n"
            f"contrast_phase: {self.contrast_phase}\n"
            f"kilovoltage peak: {self.kilo_voltage_peak}\n"
            f"mean CTDIvol: {self.mean_ctdi_vol}\n"
            f"dose length product: {self.dose_length_product}\n"
        )
        print(msg)

        if print_additional_tags:
            print(f"additional DICOM tags:\n{self.additional_tags}\n")


@dataclass
class SeriesData:
    series_inst_uid: str = None
    series_descriptions: str = None
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
    pseudoname: str = None
    study_inst_uid: str = None
    study_date: str = None
    series_dict: dict[str, SeriesData] = None
    number_of_ct_scans: int = None


def preprocess_dicom(
    input_dir: Union[str, Path], output_dir: Union[str, Path] = "./inputs"
):
    print("preprocessing dicom files")

    # check if all additional tags are valid

    if not isinstance(input_dir, Path):
        input_dir = Path(input_dir)

    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)

    # prepare table to output dicom tags
    # df_cols = [
    #     "PatientID",
    #     "Pseudoname",
    #     "StudyInstanceUID",
    #     "StudyDate",
    #     "SeriesInstanceUID",
    #     "SeriesDescription",
    #     "SliceThickness",
    #     "HasContrast",
    #     "ContrastPhase",
    #     "KiloVoltagePeak",
    # ]

    # patient_tags_df_path = Path(output_dir, "sarco_patients_dicom_tags.csv")
    # if not patient_tags_df_path.exists():
    #     patient_tags_df = pd.DataFrame([], columns=df_cols)
    #     patient_tags_df.to_csv(patient_tags_df_path, columns=df_cols, header=True, index=True, index_label="index")

    i = 1
    for dicom_dir in input_dir.glob("*/"):
        dur = perf_counter()
        dicom_files = find_dicoms(dicom_dir)

        study_data, files_by_series_uid, dose_report_path = sort_files_by_series_uid(
            dicom_files
        )

        dose_per_event = extract_dose_values(dose_report_path)

        series_to_segment = filter_series_to_segment(
            files_by_series_uid, dose_report=dose_per_event
        )

        print(f"found {len(series_to_segment)} valid series for segmentation")
        study_data.series_dict = series_to_segment

        pseudoname = f"sarco_{i}"
        i += 1
        output_case_dir = Path(output_dir, pseudoname)
        output_case_dir.mkdir(exist_ok=True, parents=True)

        for data in series_to_segment.values():
            # dose = dose_per_event.get(data.irradiation_event_uid)
            # data.dose_length_product = dose.get("dlp", None)
            # data.mean_ctdi_vol = dose.get("mean_ctdi_vol", None)

            print(
                f"saving id `{study_data.patient_id}` contrast `{data.contrast_phase}`"
            )

            dicom_datasets = [pydicom.dcmread(file) for file in data.filepaths]

            output_filepath = output_case_dir.joinpath(f"{data.contrast_phase}.nii.gz")

            try:
                dicom_array_to_nifti(
                    dicom_list=dicom_datasets,
                    output_file=output_filepath,
                    reorient_nifti=True,
                )
                print(
                    (
                        f"id `{study_data.patient_id}` (pseudoname `{pseudoname}`), "
                        f"contrast `{data.contrast_phase}`, "
                        f"{data.num_of_filepaths} DICOM files written as nifti `{output_filepath.name}`"
                    )
                )
                print(f"finished in {perf_counter() - dur}")
                print("\n" + "-" * 25)
            except RuntimeError as err:
                print(err)


def sort_files_by_series_uid(
    dicom_files: list[Path],
) -> tuple[StudyData, dict[str, list[Path]], Path]:
    """
    Sorts filepaths by DICOM tag SeriesInstanceUID.
    Also returns fullpath of Dose report series if found.

    Args:
        dicom_files (list[Path]): List of dicom files, excluding DICOMDIR file.

    Returns:
        tuple:
            StudyData: Object containing basic study metadata.
            dict[str, list[Path]]: Dictionary mapping each SeriesInstanceUID to a list of filepaths.
            Path: Path to Dose Report series file, if found, otherwise None.
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
                "SeriesInstanceUID",
                "SeriesDescription",
                "Modality",
            ],
        )
        series_uid = ds.SeriesInstanceUID

        if series_uid in files_by_uid:
            files_by_uid[series_uid].append(file)
        else:
            files_by_uid[series_uid] = [file]

        if "dose report" in ds.SeriesDescription.lower() or ds.Modality == "SR":
            dose_report_file = file

    print(
        f"\nid '{study_data.patient_id}' has {len(dicom_files)} files across {len(files_by_uid.keys())} series"
    )
    return study_data, files_by_uid, dose_report_file


def filter_series_to_segment(
    all_series: dict,
    dose_report: dict,
) -> dict[str, SeriesData]:
    """
    Filter series by checking DICOM tags. Finds the native CT series with the lowest slice thickness and highest number of slices.
    Also finds the contrast phase CT series with the lowest slice thickness and highest number of slices per each contrast phase found.

    Args:
        dataset (pydicom.FileDataset): all series found.

    Returns:
        list: list consisting of SeriesMetadata dataclass for native CT and each contract phase CT.
    """

    series_by_contrast: dict[str, list[SeriesData]] = {
        # "native": [],
        # "arterial": [],
        # "nephro": [],
        # "venous": [],
    }

    for series_uid, filepaths in all_series.items():
        # read only the first file to filter series
        # first_filepath = filepaths[0]
        dataset = pydicom.dcmread(filepaths[0], stop_before_pixels=True)

        # filter by words in SeriesDescription
        series_desc: str = dataset.SeriesDescription
        if SERIES_DESC_PATTERN.search(series_desc):
            continue

        slice_thickness = (
            float(dataset.SliceThickness)
            if hasattr(dataset, "SliceThickness")
            else None
        )
        if slice_thickness is None:
            continue

        contrast_applied = dataset.get("ContrastBolusAgent", None)

        series_data = SeriesData(
            series_inst_uid=series_uid,
            series_descriptions=series_desc,
            filepaths=filepaths,
            num_of_filepaths=len(filepaths),
            has_contrast="yes" if contrast_applied else "no",
            irradiation_event_uid=dataset.get("IrradiationEventUID", "none"),
        )

        contrast_match = CONTRAST_PHASES.search(series_desc)
        if contrast_match:
            series_data.contrast_phase = contrast_match.group().lower()
        else:
            series_data.contrast_phase = "native"
        # series_data = SeriesMetadata(
        #     patient_id=ds.PatientID,
        #     study_instance_uid=ds.StudyInstanceUID,
        #     study_date=ds.StudyDate,
        #     series_instance_uid=ds.SeriesInstanceUID,
        #     series_description=ds.SeriesDescription,
        #     filepaths=filepaths,
        #     num_of_files=len(filepaths),
        #     slice_thickness=slice_thickness,
        #     has_contrast=True if contrast_applied else False,
        #     kilo_voltage_peak=int(ds.get("KVP", None)),
        #     irradiation_event_uid=ds.IrradiationEventUID,
        # )

        # maps series data by contrast phase - "native", "arterial", ...
        # series_by_contrast[series_data.contrast_phase].append(series_data)

        series_data.kilo_voltage_peak = dataset.get("KVP", 0.0)
        tube_currents = [
            pydicom.dcmread(
                p, stop_before_pixels=True, specific_tags=["XRayTubeCurrent"]
            ).get("XRayTubeCurrent", None)
            for p in filepaths
        ]
        series_data.mean_tube_current = mean(
            [current for current in tube_currents if current]
        )

        series_data.mean_ctdi_vol = dose_report[series_data.irradiation_event_uid][
            "mean_ctdi_vol"
        ]
        series_data.dose_length_product = dose_report[
            series_data.irradiation_event_uid
        ]["dlp"]

        if series_data.contrast_phase in series_by_contrast:
            series_by_contrast[series_data.contrast_phase].append(series_data)
        else:
            series_by_contrast[series_data.contrast_phase] = [series_data]

    # selects series with lowest slice thickness and highest number of files
    # selection per contrast phase, skips empty lists
    selected_series = {
        phase: min(
            data_list, key=lambda data: (data.slice_thickness, -data.num_of_filepaths)
        )
        for phase, data_list in series_by_contrast.items()
    }

    return selected_series


def write_dicom_tags(
    path_df_dicom_tags: Union[Path, str], data: SeriesMetadata, add_dicom_tags: list
) -> int:
    df_dicom_tags = pd.read_csv(path_df_dicom_tags, header=0, index_col=0)

    last_row_index = len(df_dicom_tags)

    # start patient indexes at 1
    pseudoname = f"sarco_{last_row_index + 1}"

    """
    cols: [
        "PatientID",
        "Pseudoname",
        "Filename",
        "StudyInstanceUID",
        "StudyDate",
        "SeriesDescription",
        "SliceThickness",
        "HasContrast",
        "ContrastPhase",
        "KiloVoltagePeak"
        ] + additional_dicom_tags
    """
    row_data = [
        data.patient_id,
        pseudoname,
        f"{data.contrast_phase}.nii.gz",
        data.study_instance_uid,
        data.study_date,
        data.series_description,
        data.slice_thickness,
        data.has_contrast,
        data.contrast_phase,
        data.kilo_voltage_peak,
    ]

    if add_dicom_tags:
        row_data.extend([data.additional_tags[tag] for tag in add_dicom_tags])

    df_dicom_tags.loc[last_row_index] = row_data
    df_dicom_tags.to_csv(
        path_df_dicom_tags,
        columns=df_dicom_tags.columns,
        header=True,
        index=True,
        index_label="index",
    )
    print(
        f"id '{data.patient_id}' (pseudoname '{pseudoname}') contrast_phase '{data.contrast_phase}' written to csv"
    )
    return pseudoname


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


if __name__ == "__main__":
    if len(sys.argv) < 1:
        print("Missing input directory")
        sys.exit(-1)
    preprocess_dicom(sys.argv[1])
