import enum
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Self

import pandas as pd
from nibabel import Nifti1Image

from src.io import read_json

log = logging.getLogger("classes")


@dataclass
class SeriesData:
    series_inst_uid: str
    series_description: str | None = None
    filepaths: list[Path] = field(default_factory=list, repr=False)
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

    @classmethod
    def _from_dict(cls, series_data: dict[str, Any]) -> Self:
        return cls(**series_data)

    def _to_dict(self, exclude_fields: list[str] | None = None) -> dict[str, Any]:
        exclude = {"filepaths", "filepaths_num"}
        if exclude_fields:
            exclude.update(exclude_fields)

        return asdict(
            self,
            dict_factory=lambda dic: {
                key: val for key, val in dic if key not in exclude
            },
        )


@dataclass
class StudyData:
    participant: str
    study_inst_uid: str
    patient_id: str | None = field(default=None, repr=False, compare=False)
    # study_date: str | None = field(default=None, repr=False, compare=False)
    patient_height: float | int | None = field(default=None, repr=False, compare=False)
    series: dict[str, SeriesData] = field(default_factory=dict)

    @classmethod
    def _from_labkey_row(cls, row: dict[str, Any]) -> Self:
        return cls(
            participant=row.get("PARTICIPANT"),
            study_inst_uid=row.get("STUDY_INSTANCE_UID"),
            patient_id=row.get("RODNE_CISLO"),
            patient_height=row.get("VYSKA_PAC."),
        )

    @classmethod
    def _from_json(cls, path: str | Path) -> Self:
        """Deserialize study and series data from JSON to `StudyData`.

        Args:
            filepath (str | Path): Path to .json file.

        Returns:
            study_data (StudyData): Deserialized `StudyData` dataclass object.
        """
        data = read_json(path)
        series_dict: dict[str, Any] = data.pop("series")
        return cls(
            **data,
            series={
                key: SeriesData._from_dict(val) for key, val in series_dict.items()
            },
        )

    def get_series(self, series_uid: str) -> SeriesData | None:
        return self.series.get(series_uid)

    def _write_to_json(
        self,
        output_dir: str | Path,
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
            log.debug(f"overwriting existing file at `{str(filepath)}`")

        with open(filepath, mode="w", encoding="utf-8") as file:
            json.dump(serialized, file, indent=2)

        log.debug(
            f"written DICOM tags for {self.participant=}, {self.study_inst_uid=}\nfields excluded: {exclude}"
        )

    def _to_list_of_dicts(self) -> list[dict[str, Any]]:
        _study = {
            "participant": self.participant,
            "study_inst_uid": self.study_inst_uid,
            "patient_height": self.patient_height,
            # "study_date": self.study_date,
        }
        return [_study | series._to_dict() for series in self.series.values()]


@dataclass
class ImageData:
    image: Nifti1Image
    path: Path


@dataclass
class Centroids:
    vertebre_centroid: list = field(default_factory=list)
    body_centroid: list = field(default_factory=list)


@dataclass
class ProcessDurations:
    spine_seg: int | float = 0.0
    tissue_seg: int | float = 0.0
    slice_extraction: int | float = 0.0
    postprocessing: int | float = 0.0


@dataclass
class MetricsData:
    area: dict[str, Any]
    mean_hu: dict[str, Any]
    skelet_muscle_index: float | None = 0.0

    series_inst_uid: str | None = None
    contrast_phase: str | None = None

    process_durations: ProcessDurations | None = field(default_factory=ProcessDurations)
    total_duration: float | None = 0.0
    centroids: Centroids = field(default_factory=Centroids)

    def _to_dict(self) -> dict[str, Any]:
        tissue_labels = self.area.keys()
        return (
            {
                "series_inst_uid": self.series_inst_uid,
                "contrast_phase": self.contrast_phase,
                "skelet_muscle_index": self.skelet_muscle_index,
                "total_duration": self.total_duration,
            }
            | {f"area_{label}": self.area[label] for label in tissue_labels}
            | {f"mean_hu_{label}": self.mean_hu[label] for label in tissue_labels}
        )

    def set_durations(self, durations: ProcessDurations):
        self.process_durations = durations
        self.total_duration = sum(durations.__dict__.values())

    @classmethod
    def _from_dict(cls, d: dict[str, Any]) -> Self:
        return cls(
            area=d.get("area", {}),
            mean_hu=d.get("mean_hu", {}),
            skelet_muscle_index=d.get("skelet_muscle_index"),
            series_inst_uid=d.get("series_inst_uid"),
            contrast_phase=d.get("contrast_phase"),
            process_durations=d.get("process_durations"),
            total_duration=d.get("total_duration"),
            centroids=Centroids(
                d.get("vertebre_centroid", []), d.get("body_centroid", [])
            ),
        )


class ProcessResult(enum.Enum):
    MISSING_L3_MASK = "MISSING_L3_MASK"
    SEGMENTATION_FINISHED = "SEGMENTATION_FINISHED"
    MISSING_ON_PACS_OR_LOCAL = "MISSING_ON_PACS_OR_LOCAl"
    NO_SERIES_TO_SEGMENT = "NO_SERIES_TO_SEGMENT"


@dataclass
class SegmentationResult:
    participant: str
    study_inst_uid: str
    patient_height: float | None = None
    metrics_dict: dict[str, MetricsData] = field(default_factory=dict)
    series_process_result: dict[str, ProcessResult | str] = field(default_factory=dict)

    @classmethod
    def _from_study_case(cls, study_data: StudyData) -> Self:
        return cls(
            participant=study_data.participant,
            study_inst_uid=study_data.study_inst_uid,
            patient_height=study_data.patient_height,
        )

    # TODO: _write_to_json() maybe add exclude_fields: list[str]
    def _write_to_json(self, output_dir: str | Path):
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)

        # TODO: if needed add fields to exclude and its default
        # exclude = {}
        # if exclude_fields:
        #     exclude.update(exclude_fields)

        # TODO: temporary fix for unserializable ProcessResult
        self.series_process_result = {
            uid: value.value for uid, value in self.series_process_result.items()
        }

        serialized = asdict(
            self,
            dict_factory=lambda dic: {key: val for key, val in dic},
        )
        # if key not in exclude

        filepath = output_dir.joinpath(f"metrics_{self.study_inst_uid}.json")
        if filepath.exists():
            log.debug(f"overwriting existing file at `{str(filepath)}`")

        with open(filepath, mode="w", encoding="utf-8") as file:
            json.dump(serialized, file, indent=2)

        log.debug(
            f"written DICOM tags for participant {self.participant}, study {self.study_inst_uid}"
        )

    @classmethod
    def _from_json(cls, path: Path | str) -> Self:
        data = read_json(path)
        return cls(
            participant=data.get("participant"),
            study_inst_uid=data.get("study_inst_uid"),
            patient_height=data.get("patient_height"),
            metrics_dict={
                uid: MetricsData._from_dict(metric)
                for uid, metric in data.get("metrics_dict").items()
            },
        )

    def _to_list_of_dicts(self) -> list[dict[str, Any]]:
        _base_dict = {
            "participant": self.participant,
            "patient_height": self.patient_height,
            "study_inst_uid": self.study_inst_uid,
        }
        return [_base_dict | metric._to_dict() for metric in self.metrics_dict.values()]


@dataclass
class Report:
    timestamp: str
    data_rows: list[dict[str, Any]] = field(default_factory=list)

    def add_case(
        self,
        participant: str,
        study_instance_uid: str,
        process_result: ProcessResult,
        series_instance_uid: str | None = None,
    ):
        row: dict[str, Any] = {
            "participant": participant,
            "study_inst_uid": study_instance_uid,
            "series_inst_uid": series_instance_uid,
            "process_result": process_result.value,
        }
        self.data_rows.append(row)
        msg = f"case {participant}, study {study_instance_uid}, series {series_instance_uid}: {process_result.value}"
        if process_result in (
            ProcessResult.MISSING_L3_MASK,
            ProcessResult.MISSING_ON_PACS_OR_LOCAL,
            ProcessResult.NO_SERIES_TO_SEGMENT,
        ):
            log.warning(msg)
        else:
            log.info(msg)

    def write_report(self, directory: str | Path):
        df = pd.DataFrame(self.data_rows)
        df.to_csv(Path(directory, f"./report_{self.timestamp}.csv"), index=False)
