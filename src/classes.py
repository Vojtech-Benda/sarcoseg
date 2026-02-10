import json
from pathlib import Path
from nibabel.nifti1 import Nifti1Image
from dataclasses import asdict, dataclass, field
from numpy.typing import NDArray
from typing import Any, Union
import pandas as pd
from pydicom import dcmread
from src import slogger


logger = slogger.get_logger(__name__)


@dataclass
class LabkeyRow:
    # patient_id: str
    # study_date: str
    participant: str
    study_instance_uid: str = None
    # pacs_number: str = None
    patient_height: float = None

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
    patient_id: str | None = None
    participant: str = None
    study_inst_uid: str | None = None
    study_date: str | None = None
    # patient_height: str | None = None
    series: list[SeriesData] = field(default_factory=list)

    @classmethod
    def _from_dicom_file(cls, labkey_data: LabkeyRow, dicom_file: Union[Path, str]):
        ds = dcmread(
            dicom_file,
            stop_before_pixels=True,
            specific_tags=["StudyInstanceUID", "StudyDate"],  # [TODO]: add PatientID?
        )

        return StudyData(
            participant=labkey_data.participant,
            study_inst_uid=ds.StudyInstanceUID,
            study_date=ds.StudyDate,
            # patient_height=labkey_data.patient_height,
        )

    def _write_to_json(
        self, output_dir: Union[str, Path] = None, exclude_fields: list[str] = None
    ):
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)

        exclude = {"filepaths", "filepaths_num"}
        if exclude_fields:
            exclude.update(exclude_fields)

        data = asdict(
            self,
            dict_factory=lambda dic: {
                key: val for key, val in dic if key not in exclude
            },
        )

        filepath = output_dir.joinpath(f"dicom_tags_{self.study_inst_uid}.json")
        if filepath.exists():
            logger.info(f"overwriting existing file at `{str(filepath)}`")

        with open(filepath, mode="w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)

        logger.info(
            f"written DICOM tags for participant {self.participant}, study instance uid {self.study_inst_uid}\nfields excluded: {exclude}"
        )

    def _to_list_of_dicts(self):
        return [
            {
                "participant": self.participant,
                "study_inst_uid": self.study_inst_uid,
                "study_date": self.study_date,
            }
            | series.__dict__
            for series in self.series
        ]


@dataclass
class ImageData:
    image: Union[Nifti1Image, NDArray] = None
    spacing: NDArray = None
    path: Path = None


@dataclass
class Centroids:
    vertebre_centroid: NDArray = None
    body_centroid: NDArray = None


@dataclass
class MetricsData:
    patient_data: dict[str, Any] = field(default_factory=dict)

    area: dict[str, Any] = field(default_factory=dict)
    mean_hu: dict[str, Any] = field(default_factory=dict)
    skelet_muscle_index: float = None

    duration: float = None
    centroids: Centroids = None

    def set_patient_data(self, df_patient_data: pd.DataFrame, series_inst_uid: str):
        self.patient_data = (
            df_patient_data.loc[df_patient_data["series_inst_uid"] == series_inst_uid]
            .iloc[0]
            .to_dict()
        )

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
