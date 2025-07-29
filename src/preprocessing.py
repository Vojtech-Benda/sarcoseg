import sys
import re

from pathlib import Path
from typing import Union, Optional
from dataclasses import dataclass

# image processing packages
import pydicom
from pydicom import datadict
import pandas as pd
from dicom2nifti.common import sort_dicoms, validate_orientation
from dicom2nifti.convert_dicom import dicom_array_to_nifti
import time


SERIES_DESC_PATTERN = re.compile(r'|'.join(
    ("protocol", "topogram", "scout", "patient", "dose", "report")
    ), re.IGNORECASE)
CONTRAST_PHASES = re.compile(r'|'.join(
    ("arterial", "nephro", "venous")
    ), re.IGNORECASE)


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
            )
        print(msg)
        
        if print_additional_tags:
            print(f"additional DICOM tags:\n{self.additional_tags}\n")


def sort_files_by_series_uid(root_dir: Path, filepaths: list[str]) -> dict[str, list[str]]:
    files_by_uid: dict[str, list[str]] = {}
    
    first_file = pydicom.dcmread(Path(root_dir, filepaths[0]), stop_before_pixels=True)
    patient_id = first_file.PatientID
    
    for path in filepaths:
        dicom_path = Path(root_dir, path)
        dataset = pydicom.dcmread(dicom_path, stop_before_pixels=True)
        series_uid = dataset.SeriesInstanceUID
        
        if series_uid in files_by_uid:
            files_by_uid[series_uid].append(dicom_path)
        else:
            files_by_uid[series_uid] = [dicom_path]
    
    print(f"\nid '{patient_id}' has {len(filepaths)} files across {len(files_by_uid.keys())} series")
    return files_by_uid


def filter_series_to_segment(all_series: dict, 
                             additional_dicom_tags: Optional[list[str]] = None) -> Union[list[SeriesMetadata], None]:
    """
    Filter series by checking DICOM tags. Finds the native CT series with the lowest slice thickness and highest number of slices.
    Also finds the contrast phase CT series with the lowest slice thickness and highest number of slices per each contrast phase found.
    
    Args:
        dataset (pydicom.FileDataset): all series found.
    
    Returns:
        list: list consisting of SeriesMetadata dataclass for native CT and each contract phase CT.
    """
    
    contrast_series: dict[str, list[SeriesMetadata]] = {
        'arterial': [],
        'nephro': [],
        'venous': []
        }
    native_series = []
    
    for _, filepaths in all_series.items():
        
        # read only the first file to filter series
        # first_filepath = Path(root_dir, filepaths[0])
        first_filepath = filepaths[0]
        if not first_filepath.exists():
            raise FileNotFoundError(2, "file not found", first_filepath)
        dataset = pydicom.dcmread(first_filepath, stop_before_pixels=True)
        
        # filter by words in SeriesDescription
        series_desc: str = dataset.SeriesDescription
        if SERIES_DESC_PATTERN.search(series_desc):
            continue
        
        
        slice_thickness = float(dataset.SliceThickness) if hasattr(dataset, "SliceThickness") else None
        if slice_thickness is None:
            continue
        
        contrast_applied = dataset.get("ContrastBolusAgent", None)

        series_data = SeriesMetadata(
            patient_id=dataset.PatientID,
            study_instance_uid=dataset.StudyInstanceUID,
            study_date=dataset.StudyDate,
            series_instance_uid=dataset.SeriesInstanceUID,
            series_description=dataset.SeriesDescription,
            filepaths=filepaths,
            num_of_files=len(filepaths),
            slice_thickness=slice_thickness,
            has_contrast=True if contrast_applied else False,
            kilo_voltage_peak=int(dataset.get("KVP", None))
            )
        
        # contrast_phase = series_desc.split(" ")[0].lower()
        # contrast_phase = series_desc
        contrast_match = CONTRAST_PHASES.search(series_desc)
        if contrast_match:
            series_data.contrast_phase = contrast_match.group().lower()
        else:
            series_data.contrast_phase = "none"
        
        # add additional tags for extraction
        if additional_dicom_tags is not None:
            for tag in additional_dicom_tags:
                tag_value = dataset.get(tag, None)
                series_data.additional_tags[tag] = tag_value
        
        # separate into contrast and no contrast
        if series_data.has_contrast and series_data.contrast_phase != "none":
            contrast_series[series_data.contrast_phase].append(series_data)
        else:
            native_series.append(series_data)
            
    # select the series with lowest slice thickness AND highest number of slices
    selected_native_series = min(native_series, key=lambda v: (v.slice_thickness, -v.num_of_files))
    
    # same thing but for each contrast phase key - {'arterial': [...], 'nephro': [...], 'venous': [...]}
    # skips empty lists
    selected_contrast_series_by_phase = {
        phase: min(series_list, key=lambda v: (v.slice_thickness, v.num_of_files))
        for phase, series_list in contrast_series.items() if series_list
    }

    # return as a single list of selected series to segment
    return [selected_native_series] + list(selected_contrast_series_by_phase.values())
    

def write_dicom_tags(path_df_dicom_tags: Union[Path, str], data: SeriesMetadata, add_dicom_tags: list) -> int:
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
        f"{pseudoname}.nii.gz",
        data.study_instance_uid,
        data.study_date,
        data.series_description,
        data.slice_thickness,
        data.has_contrast,
        data.contrast_phase,
        data.kilo_voltage_peak
    ]
    
    if add_dicom_tags:
        row_data.extend([data.additional_tags[tag] for tag in add_dicom_tags])
    
    df_dicom_tags.loc[last_row_index] = row_data
    df_dicom_tags.to_csv(path_df_dicom_tags, columns=df_dicom_tags.columns, header=True, index=True, index_label="index")
    print(f"id '{data.patient_id}' (pseudoname '{pseudoname}') contrast_phase '{data.contrast_phase}' written to csv")
    return pseudoname


def preprocess_dicom(input_dir: Union[str, Path], output_dir: Union[str, Path] = "./inputs", **kwargs):
    print("preprocessing dicom files")
    
    # check if all additional tags are valid
    additional_dicom_tags = kwargs.get("dicom_tags", None)
    if additional_dicom_tags:
        invalid_tags = [tag for tag in additional_dicom_tags if not datadict.dictionary_has_tag(tag)]
        if invalid_tags:
            print(f"wrong dicom tag name(s): {', '.join(map(str, invalid_tags))}")
            sys.exit(-1)
        
    anonymize = kwargs.get("anonymize", False)
        
    input_dir = Path(input_dir)

    # prepare table to output dicom tags
    df_cols = [
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
        ]
    
    if additional_dicom_tags:
        df_cols.extend(additional_dicom_tags)
    
    patient_tags_df_path = Path(output_dir, "sarco_patients_dicom_tags.csv")
    if not patient_tags_df_path.exists():
        patient_tags_df = pd.DataFrame([], columns=df_cols)
        patient_tags_df.to_csv(patient_tags_df_path, columns=df_cols, header=True, index=True, index_label="index")
    
    for root, _, files in input_dir.walk():
        if not files or "DICOMDIR" in files:
            continue
        
        files_by_series_uid = sort_files_by_series_uid(root, files)
        
        series_to_segment = filter_series_to_segment(files_by_series_uid,  
                                                     additional_dicom_tags=additional_dicom_tags)
        print(f"found {len(series_to_segment)} valid series for segmentation")
        
        for data in series_to_segment:
            dicom_datasets = [pydicom.dcmread(file) for file in data.filepaths]
            dicom_datasets = sort_dicoms(dicom_datasets)
            
            validate_orientation(dicom_datasets)
            
            pseudoname = write_dicom_tags(patient_tags_df_path, 
                                          data, 
                                          additional_dicom_tags)
            
            nifti_filename = f"{pseudoname}_{data.contrast_phase}" if data.has_contrast else pseudoname
            output_path = Path(output_dir, f"{nifti_filename}.nii.gz")
            
            try:
                dicom_array_to_nifti(dicom_list=dicom_datasets,
                                    output_file=output_path,
                                    reorient_nifti=True)
                print(f"id '{data.patient_id}' (pseudoname '{pseudoname}') with contrast phase '{data.contrast_phase}' written to nifti as '{output_path.name}'")
            except RuntimeError as err:
                print(err)
            
if __name__ == "__main__":
    if len(sys.argv) < 1:
        print("Missing input directory")
        sys.exit(-1)
    preprocess_dicom(sys.argv[1])