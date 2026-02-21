import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Self, Union

import pandas as pd
from nibabel.nifti1 import Nifti1Image
from numpy.typing import NDArray
from pydicom import dcmread

from src import slogger

logger = slogger.get_logger(__name__)


@dataclass
class LabkeyRow:
    row_id: str
    # patient_id: str
    # study_date: str
    participant: str
    study_instance_uid: str | None = None
    # pacs_number: str = None
    patient_height: float | None = None

    @classmethod
    def from_labkey_dict(cls, row: dict):
        return cls(
            # [TODO]: add later if needed for PACS C-MOVE by id and date, may cause issue with private information
            # study_date=row.get("CAS_VYSETRENI").split(" ")[
            #     0
            # ],  # take date, discard time ["date", "time"] OR take both
            participant=row.get("PARTICIPANT"),
            study_instance_uid=row.get("STUDY_INSTANCE_UID"),
            # [TODO]: maybe C-MOVE by PACS_CISLO (DICOM tag is AccessionNumber), but some records may be missing PACS_NUMBER
            # pacs_number=row.get("PACS_CISLO"),
            patient_height=row.get("VYSKA_PAC."),
        )


@dataclass
class SeriesData:
    series_inst_uid: str | None = None
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
    participant: str
    row_id: str | None = field(default=None, compare=False, repr=False)
    study_inst_uid: str | None = field(default=None, compare=False)
    patient_id: str | None = field(default=None, repr=False, compare=False)
    study_date: str | None = field(default=None, repr=False, compare=False)
    patient_height: float | int | None = field(default=None, repr=False, compare=False)
    series: dict[str, SeriesData] = field(default_factory=dict)

    """
    @classmethod
    def _from_dicom_file(
        cls, labkey_data: LabkeyRow, dicom_file: Union[Path, str]
    ) -> Self:
        ds = dcmread(
            dicom_file,
            stop_before_pixels=True,
            specific_tags=["StudyInstanceUID", "StudyDate"],  # [TODO]: add PatientID?
        )

        return StudyData(
            participant=labkey_data.participant,
            study_inst_uid=ds.StudyInstanceUID,
            study_date=ds.StudyDate,
        )
    """

    @classmethod
    def from_labkey_row(cls, row: dict[str, Any]) -> Self:
        return cls(
            participant=row.get("PARTICIPANT"),
            row_id=row.get("ID"),
            study_inst_uid=row.get("STUDY_INSTANCE_UID"),
            patient_id=row.get("RODNE_CISLO"),
            patient_height=row.get("VYSKA_PAC."),
        )

    def get_series(self, series_uid: str) -> Union[SeriesData, None]:
        return self.series.get(series_uid)

    def _write_to_json(
        self,
        output_dir: Union[str, Path],
        exclude_fields: list[str] | None = None,
    ):
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)

        exclude = {"filepaths", "filepaths_num"}
        if exclude_fields:
            exclude.update(exclude_fields)

        serialized = asdict(
            self,
            dict_factory=lambda dic: {
                key: val for key, val in dic if key not in exclude
            },
        )

        filepath = output_dir.joinpath(f"dicom_tags_{self.study_inst_uid}.json")
        if filepath.exists():
            logger.info(f"overwriting existing file at `{str(filepath)}`")

        with open(filepath, mode="w", encoding="utf-8") as file:
            json.dump(serialized, file, indent=2)

        logger.info(
            f"written DICOM tags for participant {self.participant}, study instance uid {self.study_inst_uid}\nfields excluded: {exclude}"
        )

    def _to_list_of_dicts(self):
        _study = {
            "participant": self.participant,
            "study_inst_uid": self.study_inst_uid,
            "study_date": self.study_date,
        }
        return [_study | series.__dict__ for series in self.series.values()]


@dataclass
class ImageData:
    image: Union[Nifti1Image, NDArray]
    path: Path
    spacing: Optional[NDArray] = None


@dataclass
class Centroids:
    vertebre_centroid: list
    body_centroid: list


@dataclass
class MetricsData:
    area: dict[str, Any]
    mean_hu: dict[str, Any]
    skelet_muscle_index: Optional[float] = None

    series_inst_uid: Optional[str] = None
    contrast_phase: str | None = None

    duration: Optional[float] = None
    centroids: Optional[Centroids] = None

    def _to_dict(self):
        row = {}
        row.update(self.patient_data)
        row.update({f"area_{k}": v for k, v in self.area.items()})
        row.update({f"mean_hu_{k}": v for k, v in self.mean_hu.items()})
        row["skelet_muscle_index"] = self.skelet_muscle_index
        row["duration"] = self.duration
        row["vertebra_centroid_slice"] = self.centroids.vertebre_centroid[-1]
        row["vertebra_body_centroid_slice"] = self.centroids.body_centroid[-1]
        return row

    def set_duration(self, *durations):
        self.duration = sum(durations)


@dataclass
class SegmentationResult:
    participant: str
    study_inst_uid: str
    patient_height: str | None = None

    metrics_dict: dict[str, MetricsData] = field(default_factory=dict)

    @classmethod
    def _from_study_case(cls, study_data: StudyData):
        return SegmentationResult(
            participant=study_data.participant,
            study_inst_uid=study_data.study_inst_uid,
            patient_height=study_data.patient_height,
        )

    def _write_to_json(self, output_dir: Union[str, Path], exclude_fields: list[str]):
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)

        # [TODO]: if needed add fields to exclude and its default
        # exclude = {}
        # if exclude_fields:
        #     exclude.update(exclude_fields)

        serialized = asdict(
            self,
            dict_factory=lambda dic: {key: val for key, val in dic},
        )
        # if key not in exclude

        filepath = output_dir.joinpath(f"metrics_{self.study_inst_uid}.json")
        if filepath.exists():
            logger.info(f"overwriting existing file at `{str(filepath)}`")

        with open(filepath, mode="w", encoding="utf-8") as file:
            json.dump(serialized, file, indent=2)

        logger.info(
            f"written DICOM tags for participant {self.participant}, study instance uid {self.study_inst_uid}"
        )
