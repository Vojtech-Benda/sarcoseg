from pathlib import Path
from nibabel.nifti1 import Nifti1Image
from dataclasses import dataclass, field
from numpy.typing import NDArray
from typing import Any, Union
import pandas as pd


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


@dataclass
class LabkeyData:
    data: dict[str, Any] = field(default_factory=dict)
    query_columns: list[str] = field(default_factory=list)


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
