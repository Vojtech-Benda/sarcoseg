import sys
import re
import shutil

from pathlib import Path
from typing import Any, Union

import pandas as pd
import pydicom
import dcm2niix
from statistics import mean
from datetime import datetime

# from src.network import database
from src.classes import SeriesData, StudyData
from src.network.database import LabkeyRow
from src import slogger


logger = slogger.get_logger(__name__)


SERIES_DESC_PATTERN = re.compile(
    r"|".join(
        ("protocol", "topogram", "scout", "patient", "dose", "report", "monitor")
    ),
    re.IGNORECASE,
)
CONTRAST_PHASES_PATTERN = re.compile(
    r"|".join(("abdomen", "arterial", "nephro", "venous")), re.IGNORECASE
)


def preprocess_dicom_study(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path] = Path("./inputs"),
    labkey_data: dict = None,
) -> StudyData:
    logger.info("preprocessing DICOM files")

    if isinstance(input_dir, str):
        input_dir = Path(input_dir)

    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    dicom_files = find_dicoms(input_dir)

    if not dicom_files:
        logger.error(f"no DICOM files found in `{input_dir}`")
        return None

    study_data, files_by_series_uid, dose_report_path = filter_dicom_files(dicom_files)

    dose_per_event = extract_dose_values(dose_report_path)

    if not dose_per_event:
        logger.warning(
            f"PatientID {study_data.patient_id}, study {study_data.study_inst_uid} does not have Dose Report"
        )
        logger.warning(f"{dose_per_event=}")

    study_data.series_dict = select_series_to_segment(
        files_by_series_uid, dose_report=dose_per_event
    )

    logger.info(f"found {len(study_data.series_dict)} valid series for segmentation")
    logger.info(
        f"saving ID {study_data.patient_id}, study instance uid `{study_data.study_inst_uid}`"
    )

    output_dir.mkdir(exist_ok=True, parents=True)

    write_dicom_tags(output_dir, study_data, labkey_data)

    for series_data in study_data.series_dict.values():
        write_series_as_nifti(output_dir, series_data)

    logger.info("-" * 25)
    return study_data


def write_series_as_nifti(output_study_dir: Path, series_data: SeriesData):
    logger.info(
        f"series instance uid `{series_data.series_inst_uid}`\n"
        f"series description `{series_data.series_description}`\n"
        f"contrast phase `{series_data.has_contrast}`, type `{series_data.contrast_phase}`"
    )

    output_series_dir = output_study_dir.joinpath(series_data.series_inst_uid)
    output_series_dir.mkdir(exist_ok=True, parents=True)
    output_filepath = output_series_dir.joinpath("input_ct_volume.nii.gz")

    tmp_dir = Path(output_study_dir, f"tmp_{series_data.series_inst_uid}")
    tmp_dir.mkdir(exist_ok=True, parents=True)
    [shutil.copy2(file, tmp_dir / file.name) for file in series_data.filepaths]

    if output_filepath.exists():
        logger.info(
            f"overwriting existing input_ct_volume.nii.gz at `{str(output_filepath.parent)}`"
        )

    try:
        args = [
            "-o",
            str(output_series_dir),
            "-f",
            "input_ct_volume",
            "-z",
            "y",
            "-b",
            "n",
            "-w",
            "1",
            str(tmp_dir),
        ]
        returncode = dcm2niix.main(args, capture_output=True, text=True)
        shutil.rmtree(tmp_dir)
    except RuntimeError as err:
        logger.error(err)
    logger.info(f"finished NifTI conversion with {returncode=}\n")


def filter_dicom_files(
    dicom_files: list[Path],
) -> tuple[StudyData, dict[str, list[Path]], Path]:
    """
    Sorts filepaths by DICOM tag SeriesInstanceUID and removes (filters) out files matching these rules:
    - SeriesDescription contains "protocol", "topogram", "scout", "patient", "dose", "report", "monitor"
        - excluding dose report
    - SliceThickness is None
    - ImageType contains "DERIVED"
    Also returns fullpath of Dose report series if found.

    Args:
        dicom_files (list[Path]): List of dicom files, excluding DICOMDIR file.

    Returns:
        StudyData: Dataclass containing basic study metadata.\n
        dict[str, list[Path]]: Dictionary mapping each SeriesInstanceUID to a list of filepaths.\n
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
                "ImageType",
                "SeriesInstanceUID",
                "SeriesDescription",
                "Modality",
                "SliceThickness",
            ],
        )

        if "dose report" in ds.SeriesDescription.lower():
            dose_report_file = file
            continue

        # filter out files in series matching pattern:
        # ("protocol", "topogram", "scout", "patient", "dose", "report"), case insensitive
        if SERIES_DESC_PATTERN.search(ds.SeriesDescription):
            continue

        if not hasattr(ds, "SliceThickness"):
            continue

        if "DERIVED" in ds.ImageType:
            continue

        series_uid = ds.SeriesInstanceUID
        if series_uid in files_by_uid:
            files_by_uid[series_uid].append(file)
        else:
            files_by_uid[series_uid] = [file]

    logger.info(
        f"\nid '{study_data.patient_id}' filtered {len(files_by_uid.keys())} series"
    )
    return study_data, files_by_uid, dose_report_file


def select_series_to_segment(
    all_series: dict[str, list[Path]],
    dose_report: dict[str, dict],
) -> dict[str, SeriesData]:
    """
    Return one or more CT series with lowest slice thickness and highest file count based on contrast phase type:
        - abdomen (native, no constrast phase), arterial, venous, nephrous.

    Args:
        all_series (dict[str, list[Path]]): series to be selected for segmentation.
        dose_report (dict[str, dict]): mapping of `IrradiationEventUID: dose_values`.

    Returns:
        dict[str, SeriesData]: mapping of `contrast_phase: SeriesData`.
    """

    series_by_contrast: dict[str, list[SeriesData]] = {}

    for series_uid, filepaths in all_series.items():
        # read only the first file to filter series
        dataset = pydicom.dcmread(filepaths[0], stop_before_pixels=True)

        # filter by words in SeriesDescription
        series_desc: str = dataset.SeriesDescription

        contrast_applied = dataset.get("ContrastBolusAgent", None)
        convolution_kernel = dataset.get("ConvolutionKernel", None)

        series_data = SeriesData(
            series_inst_uid=series_uid,
            series_description=series_desc,
            slice_thickness=float(dataset.SliceThickness),
            filepaths=filepaths,
            num_of_filepaths=len(filepaths),
            has_contrast="yes" if contrast_applied else "no",
            irradiation_event_uid=dataset.get("IrradiationEventUID", "unknown"),
            convolution_kernel=convolution_kernel[0]
            if convolution_kernel
            else "uknown",
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

        series_data.mean_ctdi_vol = dose_report.get(
            series_data.irradiation_event_uid
        ).get("mean_ctdi_vol", -1)
        series_data.dose_length_product = dose_report.get(
            series_data.irradiation_event_uid
        ).get("dlp", -1)

    return selected_series


def extract_dose_values(dose_report: str) -> dict[str, float]:
    """
    Map IrradiationEventUID to dose values

    """
    ds = pydicom.dcmread(dose_report)
    event_to_dose = {}

    if ds.Modality != "SR":
        logger.warning(f"file {dose_report} is not dose report")
        return {}

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


def find_dicoms(dicom_dir: Path) -> Union[list[Path], None]:
    """
    Returns DICOM files with matching expression `*CT*`.

    Args:
        dicom_dir (Path): Path to DICOM directory.

    Returns:
        paths (list[Path] | None): List of DICOM filepaths, otherwise `None`.
    """

    paths = [f for f in dicom_dir.rglob("*CT*") if f.is_file()]
    if not paths:
        return None
    return paths


def write_dicom_tags(study_dir: Path, study: StudyData, labkey_data: LabkeyRow = None):
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
        if labkey_data:
            row["participant"] = labkey_data.participant
            row["vyska_pac."] = labkey_data.patient_height

        rows.append(row)

    df = pd.DataFrame(rows, columns=rows[0].keys())
    filepath = study_dir.joinpath(f"dicom_tags_{study.study_inst_uid}.csv")

    if filepath.exists():
        logger.info(f"overwriting existing dicom_tags.csv at `{str(filepath)}`")

    df.to_csv(
        filepath,
        sep=",",
        na_rep="nan",
        index=False,
        columns=df.columns,
    )
    logger.info(
        f"DICOM tags for ID `{study.patient_id}` study instance uid `{study.study_inst_uid}` written to `{filepath}`"
    )


def collect_all_dicom_tags(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path] = None,
    write_to_csv: bool = False,
) -> pd.DataFrame:
    if isinstance(input_dir, str):
        input_dir = Path(input_dir)

    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    dicom_tags_files = list(input_dir.rglob("dicom_tags.*"))
    df = pd.concat(
        (pd.read_csv(file, index_col=None, header=0) for file in dicom_tags_files),
        axis=0,
        ignore_index=True,
    )

    logger.info(
        f"collected DICOM tags of {len(df.study_inst_uid.unique())} studies ({len(df.series_inst_uid.unique())} series)"
    )
    if write_to_csv:
        filepath = Path(
            output_dir if output_dir else input_dir,
            f"all_dicom_tags_{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.csv",
        )
        df.to_csv(filepath, sep=",", na_rep="nan", index=None, columns=df.columns)
        logger.info(f"DICOM tags written to `{filepath}`")

    return df


if __name__ == "__main__":
    if len(sys.argv) < 1:
        logger.critical("Missing input directory")
        sys.exit(-1)
    preprocess_dicom_study(sys.argv[1])
