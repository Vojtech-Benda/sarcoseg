import sys
import re
import os

from pathlib import Path
from typing import Union, Optional
from dataclasses import dataclass

# image processing packages
import pydicom
from pydicom import datadict
import pandas as pd
from dicom2nifti.common import sort_dicoms, validate_orientation
from dicom2nifti.convert_dicom import dicom_array_to_nifti


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


def sort_files_by_series_uid(
    root_dir: Path, filepaths: list[str]) -> dict[str, list[str]]:
    """
    Sorts filepaths by DICOM tag SeriesInstanceUID. 
    Also returns fullpath of Dose report series if found.

    Args:
        root_dir (Path): Root directory for filepaths
        filepaths (list[str]): list of filepaths

    Returns:
        dict[str, list[str]]: map of SeriesInstanceUID and sorted lists of filepaths
    """
    files_by_uid: dict[str, list[str]] = {}
    
    first_file = pydicom.dcmread(Path(root_dir, filepaths[0]), stop_before_pixels=True)
    patient_id = first_file.PatientID
    
    dose_report_path = None
    for path in filepaths:
        dicom_path = Path(root_dir, path)
        dataset = pydicom.dcmread(dicom_path, stop_before_pixels=True)
        series_uid = dataset.SeriesInstanceUID
        
        if series_uid in files_by_uid:
            files_by_uid[series_uid].append(dicom_path)
        else:
            files_by_uid[series_uid] = [dicom_path]
    
        if "dose report" in dataset.SeriesDescription.lower():
            dose_report_path = dicom_path
            print(f"found dose report file at {dose_report_path}")
    
    print(f"\nid '{patient_id}' has {len(filepaths)} files across {len(files_by_uid.keys())} series")
    return files_by_uid, dose_report_path


def filter_series_to_segment(
    all_series: dict, 
    additional_dicom_tags: Optional[list[str]] = None
    ) -> Union[list[SeriesMetadata], None]:
    """
    Filter series by checking DICOM tags. Finds the native CT series with the lowest slice thickness and highest number of slices.
    Also finds the contrast phase CT series with the lowest slice thickness and highest number of slices per each contrast phase found.
    
    Args:
        dataset (pydicom.FileDataset): all series found.
    
    Returns:
        list: list consisting of SeriesMetadata dataclass for native CT and each contract phase CT.
    """
    
    series_by_contrast: dict[str, list[SeriesMetadata]] = {
        'native': [],
        'arterial': [],
        'nephro': [],
        'venous': []
        }
    
    for _, filepaths in all_series.items():
        
        # read only the first file to filter series
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
            kilo_voltage_peak=int(dataset.get("KVP", None)),
            irradiation_event_uid=dataset.IrradiationEventUID
            )
        
        contrast_match = CONTRAST_PHASES.search(series_desc)
        if contrast_match:
            series_data.contrast_phase = contrast_match.group().lower()
        else:
            series_data.contrast_phase = "native"
        
        # add additional tags for extraction
        if additional_dicom_tags is not None:
            for tag in additional_dicom_tags:
                tag_value = dataset.get(tag, None)
                series_data.additional_tags[tag] = tag_value
        
        # maps series data by contrast phase - "native", "arterial", ...
        series_by_contrast[series_data.contrast_phase].append(series_data)
    
    # selects series with lowest slice thickness and highest number of files
    # selection per contrast phase, skips empty lists
    selected_series_by_contrast_phase = {
        phase: min(series_list, key=lambda v: (v.slice_thickness, -v.num_of_files))
        for phase, series_list in series_by_contrast.items() if series_list
    }

    return selected_series_by_contrast_phase
    

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
        f"{data.contrast_phase}.nii.gz",
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
    
    # patient_tags_df_path = Path(output_dir, "sarco_patients_dicom_tags.csv")
    # if not patient_tags_df_path.exists():
    #     patient_tags_df = pd.DataFrame([], columns=df_cols)
    #     patient_tags_df.to_csv(patient_tags_df_path, columns=df_cols, header=True, index=True, index_label="index")
    
    i = 1
    for root, _, files in input_dir.walk():
        if not files or "DICOMDIR" in files:
            continue
        
        files_by_series_uid, dose_report_path = sort_files_by_series_uid(root, files)
        
        dose_per_event = extract_dose_values(dose_report_path)
        
        series_to_segment = filter_series_to_segment(files_by_series_uid,  
                                                     additional_dicom_tags=additional_dicom_tags)
        
        print(f"found {len(series_to_segment)} valid series for segmentation")
        
        pseudoname = f"sarco_{i}"
        i += 1
        output_case_dir = Path(output_dir, pseudoname)
        os.makedirs(output_case_dir, exist_ok=True)
        
        for data in series_to_segment.values():
            dose = dose_per_event.get(data.irradiation_event_uid)
            data.dose_length_product = dose.get('dlp', None)
            data.mean_ctdi_vol = dose.get('mean_ctdi_vol', None)
            
            data.print_data()
            
            dicom_datasets = [pydicom.dcmread(file) for file in data.filepaths]
            dicom_datasets = sort_dicoms(dicom_datasets)
            
            validate_orientation(dicom_datasets)
                        
            output_filepath = output_case_dir.joinpath(f"{data.contrast_phase}.nii.gz")
            
            try:
                dicom_array_to_nifti(dicom_list=dicom_datasets,
                                    output_file=output_filepath,
                                    reorient_nifti=True)
                print(f"id '{data.patient_id}' (pseudoname '{pseudoname}') with contrast phase '{data.contrast_phase}' written to nifti '{output_filepath.name}'")
            except RuntimeError as err:
                print(err)
            
            
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
                        event_to_dose[current_event] = {'dlp': None, 'mean_ctdi_vol': None}

            # find dose values
            if vr == "NUM" and hasattr(item, "MeasuredValueSequence"):
                code_meaning = item.ConceptNameCodeSequence[0].CodeMeaning
                val = float(item.MeasuredValueSequence[0].NumericValue)
                
                if current_event:
                    if code_meaning in ["DLP"]:
                        event_to_dose[current_event]['dlp'] = val                    
                    elif code_meaning in ["Mean CTDIvol"]:
                        event_to_dose[current_event]['mean_ctdi_vol'] = val                    
                
            # recursively iterate content sequences
            if hasattr(item, "ContentSequence"):
                walk_content(item.ContentSequence, current_event)

    walk_content(ds.ContentSequence)
    return event_to_dose
            
            
            
if __name__ == "__main__":
    if len(sys.argv) < 1:
        print("Missing input directory")
        sys.exit(-1)
    preprocess_dicom(sys.argv[1])