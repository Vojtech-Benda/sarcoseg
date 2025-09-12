from pathlib import Path
from nibabel.nifti1 import Nifti1Image
from dataclasses import dataclass, field
from numpy.typing import NDArray
from typing import Any, Union
import pandas as pd


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
    area: dict[str, Any] = field(default_factory=dict)
    mean_hu: dict[str, Any] = field(default_factory=dict)
    patient_data: dict[str, Any] = field(default_factory=dict)

    duration: float = None
    centroids: Centroids = None

    def set_patient_data(self, df_patient_data: pd.DataFrame, series_inst_uid: str):
        self.patient_data = (
            df_patient_data.loc[df_patient_data["series_inst_uid"] == series_inst_uid]
            .iloc[0]
            .to_dict()
        )

    def to_dict(self):
        row = {}
        row.update(self.patient_data)
        row.update({f"area_{k}": v for k, v in self.area.items()})
        row.update({f"mean_hu_{k}": v for k, v in self.mean_hu.items()})
        row["duration"] = self.duration
        row["vertebra_centroid_height"] = self.centroids.vertebre_centroid[-1]
        row["vertebra_body_centroid_height"] = self.centroids.body_centroid[-1]
        return row

    def set_duration(self, *durations):
        self.duration = sum(durations)
