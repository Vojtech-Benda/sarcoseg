import sys
import os
import shutil
import argparse
from pathlib import Path
import SimpleITK as sitk
from typing import Any


def get_args():
    parser = argparse.ArgumentParser(prog="filter_series.py", description="filter series")
    parser.add_argument("in_directory", type=str, help="input directory containing dicom files")
    parser.add_argument("-od", "--out_directory", type=str, help="output directory of filtered studies", default="./outputs")
    return parser.parse_args()


args = get_args()
reader = sitk.ImageFileReader()

in_directory = Path(args.in_directory)

for root, _, files in os.walk(in_directory):
    if not files or "DICOMDIR" in files:
        continue
    
    filtered_series: dict[str, dict[str, Any]] = {}
    print("getting series uids")
    series_ids = sitk.ImageSeriesReader_GetGDCMSeriesIDs(root)
    
    for series_id in series_ids:
        print(f"processing uid {series_id}")
        filenames = sitk.ImageSeriesReader_GetGDCMSeriesFileNames(root, series_id)
        reader.SetFileName(filenames[0])
        reader.ReadImageInformation()
        
        patient_id = reader.GetMetaData("0010|0020")
        
        # skip series with contrast applied tag
        if reader.HasMetaDataKey("0018|0010"):
            contrast = reader.GetMetaData("0018|0010")
            if contrast:
                contrast = contrast.lower()
                if "no" not in contrast:
                    continue 
        
        if reader.HasMetaDataKey("0018|0050"):
            slice_thickness = reader.GetMetaData("0018|0050")
            if slice_thickness:
                filtered_series[series_id] = {
                    'slice_thickness': float(slice_thickness),
                    'filenames': filenames, 
                    'patient_id': patient_id
                    }
        
    selected_uid = min(filtered_series, key=lambda uid: filtered_series[uid]['slice_thickness'])
    selected_series = filtered_series[selected_uid]
    
    print(f"selected series: {selected_uid} with slice thickness: {selected_series['slice_thickness']}, files: {len(selected_series['filenames'])}")
        
    output_directory = Path(args.out_directory, selected_series['patient_id'])
    os.makedirs(output_directory)
    
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(selected_series['filenames'])
    image = series_reader.Execute()
    sitk.WriteImage(image[..., 100:200], "test.nii.gz")
    output_filepath = Path(args.out_directory, selected_series['patient_id'] + ".nii.gz")
    sitk.WriteImage(image, output_filepath, imageIO="NiftiImageIO")
    print(f"saved series image {selected_series['patient_id']}/{selected_uid} to {output_filepath}")
    
    # elif args.output_type == "dicom":
    
    #     moved_files = 0
    #     for src_path in selected_series['filenames']:
    #         src_path = Path(src_path).absolute()
    #         dest_path = Path(output_directory, src_path.stem).absolute()
            
    #         try:
    #             shutil.copy2(src=src_path, dst=dest_path)
    #             moved_files += 1
    #         except Exception as exc:
    #             print(exc)
        
    #     print(f"moved {moved_files}/{len(selected_series['filenames'])} files")
    # else:
    #     print("Unknown output data type")
    #     sys.exit(-1)