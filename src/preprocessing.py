import sys
import re
import shutil

from pathlib import Path
from typing import Any, Union
from dataclasses import dataclass, field

import pandas as pd
import pydicom
import dcm2niix
from statistics import mean
from datetime import datetime

# from src.classes import SeriesData, StudyData
from src.network.database import LabkeyRow
from src import slogger


logger = slogger.get_logger(__name__)


@dataclass
class SeriesData:
    uid: str | None = None
    description: str | None = None
    filepaths: list[Path] | None = field(default=None, repr=False)
    filepaths_num: int | None = field(default=None, repr=False)
    slice_thickness: float | None = field(default=None, repr=False)
    convolution_kernel: str | None = field(default=None, repr=False)
    has_contrast: str | None = field(default=None, repr=False)
    contrast_phase: str | None = field(default=None, repr=False)
    kilo_voltage_peak: float | None = field(default=None, repr=False)
    mean_tube_current: float | None = field(default=None, repr=False)
    irradiation_event_uid: str | None = field(default=None, repr=False)
    mean_ctdi_vol: float | None = field(default=None, repr=False)
    dose_length_product: float | None = field(default=None, repr=False)


@dataclass
class StudyData:
    patient_id: str | None = None
    uid: str | None = None
    date: str | None = None
    series: list[SeriesData] = field(default_factory=list)


class DicomStudyPreprocessor:
    def __init__(self):
        pass

    def preprocess_study(self):
        pass


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
    labkey_data: LabkeyRow | None = None,
) -> Union[StudyData, None]:
    logger.info("preprocessing DICOM files")

    if isinstance(input_dir, str):
        input_dir = Path(input_dir)

    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    dicom_files = find_dicoms(input_dir)

    if not dicom_files:
        logger.error(f"no DICOM files found in `{input_dir}`")
        return None

    study_data, series_files_map, dose_report_path = filter_dicom_files(dicom_files)

    event_to_dose = None
    if dose_report_path:
        event_to_dose = extract_dose_values(dose_report_path)

    study_data.series = select_series_to_segment(
        series_files_map, event_to_dose=event_to_dose
    )

    logger.info(f"found {len(study_data.series)} valid series for segmentation")
    logger.info(
        f"saving ID {study_data.patient_id}, study instance uid `{study_data.uid}`"
    )

    output_dir.mkdir(exist_ok=True, parents=True)

    write_dicom_tags(output_dir, study_data, labkey_data)

    for series_data in study_data.series:
        write_series_as_nifti(output_dir, series_data)

    logger.info("-" * 25)
    return study_data


def write_series_as_nifti(output_study_dir: Path, series_data: SeriesData):
    logger.info(
        f"series instance uid `{series_data.uid}`\n"
        f"series description `{series_data.description}`\n"
        f"contrast phase `{series_data.has_contrast}`, type `{series_data.contrast_phase}`"
    )

    output_series_dir = output_study_dir.joinpath(series_data.uid)
    output_series_dir.mkdir(exist_ok=True, parents=True)
    output_filepath = output_series_dir.joinpath("input_ct_volume.nii.gz")

    tmp_dir = Path(output_study_dir, f"tmp_{series_data.description}")
    tmp_dir.mkdir(exist_ok=True, parents=True)
    [shutil.copy2(file, tmp_dir.joinpath(file.name)) for file in series_data.filepaths]

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
) -> tuple[StudyData, dict[str, list[Path]], Path | None]:
    """
    Sorts filepaths by DICOM tag SeriesInstanceUID and removes (filters) out files matching these rules:
    * SeriesDescription contains `protocol`, `topogram`, `scout`, `dose`, `report`, `patient`, `monitor`
        * excluding dose report
    * SliceThickness is None
    * ImageType contains `DERIVED`

    Also tries to find and return fullpath Dose report series.

    Args:
        dicom_files (list[Path]): List of dicom files, excluding DICOMDIR file.

    Returns:
        tuple: A tuple of filtered study information.
        - **study_data** (StudyData): A dataclass containing DICOM study level tags.
        - **series_files_map** (dict[str, list[Path]]): Dictionary of filepaths mapped by SeriesInstanceUID keys.
        - **dose_report_file** (Path | None): path to Dose Report file if found, otherwise `None`.
    """

    series_files_map: dict[str, list[Path]] = {}

    ds = pydicom.dcmread(
        dicom_files[0],
        stop_before_pixels=True,
        specific_tags=["PatientID", "StudyDate", "StudyInstanceUID"],
    )

    study_data = StudyData(
        patient_id=ds.PatientID,
        uid=ds.StudyInstanceUID,
        date=ds.StudyDate,
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
        if series_uid in series_files_map:
            series_files_map[series_uid].append(file)
        else:
            series_files_map[series_uid] = [file]

    logger.info(
        f"\nid '{study_data.patient_id}' filtered {len(series_files_map.keys())} series"
    )

    if not dose_report_file:
        logger.warning(
            f"dose report DICOM file not found for PatientID {study_data.patient_id} (study {study_data.uid})"
        )

    return study_data, series_files_map, dose_report_file


def select_series_to_segment(
    series_files_map: dict[str, list[Path]],
    event_to_dose: dict[str, dict[str, float]] | None,
) -> list[SeriesData]:
    """
    Return one or more CT series with lowest slice thickness and highest file count based on contrast phase type:
        - abdomen (ie. native, no constrast phase), arterial, venous, nephrous.

    Args:
        all_series (dict[str, list[Path]]): Mapping of `SeriesInstanceUID` to filepaths.
        event_to_dose (dict[str, dict[str, float]]): Mapping of `IrradiationEventUID` to dose values.

    Returns:
        series_list (list[SeriesData]): List of series selected for segmentation.
    """

    series_by_contrast: dict[str, list[SeriesData]] = {}

    for series_uid, filepaths in series_files_map.items():
        # read only the first file to filter series
        dataset = pydicom.dcmread(filepaths[0], stop_before_pixels=True)

        # filter by words in SeriesDescription
        series_desc: str = dataset.SeriesDescription

        contrast_applied = dataset.get("ContrastBolusAgent", None)
        convolution_kernel = dataset.get("ConvolutionKernel", None)

        series_data = SeriesData(
            uid=series_uid,
            description=series_desc,
            slice_thickness=float(dataset.SliceThickness),
            filepaths=filepaths,
            filepaths_num=len(filepaths),
            has_contrast="yes" if contrast_applied else "no",
            irradiation_event_uid=dataset.get("IrradiationEventUID", "n/a"),
            convolution_kernel=convolution_kernel[0] if convolution_kernel else "n/a",
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
            key=lambda data: (data.slice_thickness, -data.filepaths_num),
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

        if event_to_dose and series_data.irradiation_event_uid != "n/a":
            series_data.mean_ctdi_vol = event_to_dose.get(
                series_data.irradiation_event_uid
            ).get("mean_ctdi_vol", -1.0)
            series_data.dose_length_product = event_to_dose.get(
                series_data.irradiation_event_uid
            ).get("dlp", -1.0)

    return list(selected_series.values())


def extract_dose_values(dose_filepath: Union[str, Path]) -> dict[str, dict[str, float]]:
    """
    Maps IrradiationEventUID to a map of dose values:
        - EventUID1
            - DLP = val
            - Mean CTDIvol = val
        - EventUID2
            - DLP = val
            - Mean CTDIvol = val
        - ...

    Args:
        dose_filepath (str | Path): Path to dose report DICOM file.

    Returns:
        event_to_dose (dict[str, dict[str, float]]): map of IrradiationEventUID to map of dose values
    """
    ds = pydicom.dcmread(dose_filepath)

    if ds.Modality != "SR":
        logger.warning(f"file {dose_filepath} is not dose report")
        return {}

    event_to_dose = {}

    def walk_sequence(sequence, current_event=None):
        for item in sequence:
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
                walk_sequence(item.ContentSequence, current_event)

    walk_sequence(ds.ContentSequence)
    return event_to_dose


def find_dicoms(dicom_dir: Path) -> Union[list[Path], None]:
    """
    Returns DICOM files with matching `*CT*`.

    Args:
        dicom_dir (Path): Path to DICOM directory.

    Returns:
        paths (list[Path] | None): List of DICOM filepaths, otherwise `None`.
    """

    for root, _, files in dicom_dir.walk():
        if len(files) == 0 or "DICOMDIR" in files:
            continue
        paths = [f for f in root.iterdir() if f.is_file()]

        if not paths:
            return None
        return paths

    # paths = [f for f in dicom_dir.rglob("*CT*") if f.is_file()]
    # if not paths:
    #     return None
    # return paths


def write_dicom_tags(
    study_dir: Path, study: StudyData, labkey_data: LabkeyRow | None = None
):
    rows: list[dict[str, Any]] = []
    for series in study.series:
        row = {
            "patient_id": study.patient_id,
            "study_inst_uid": study.uid,
            "study_date": study,
            "series_inst_uid": series.uid,
            "series_description": series.description,
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
    filepath = study_dir.joinpath(f"dicom_tags_{study.uid}.csv")

    if filepath.exists():
        logger.info(f"overwriting existing dicom_tags.csv at `{str(filepath)}`")

    df.to_csv(
        filepath,
        sep=",",
        na_rep="nan",
        index=False,
        columns=df.columns.to_list(),
    )
    logger.info(
        f"DICOM tags for ID `{study.patient_id}` study instance uid `{study.uid}` written to `{filepath}`"
    )


def collect_all_dicom_tags(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
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
        df.to_csv(
            filepath, sep=",", na_rep="nan", index=False, columns=df.columns.to_list()
        )
        logger.info(f"DICOM tags written to `{filepath}`")

    return df


if __name__ == "__main__":
    if len(sys.argv) < 1:
        logger.critical("Missing input directory")
        sys.exit(-1)
    preprocess_dicom_study(sys.argv[1])
