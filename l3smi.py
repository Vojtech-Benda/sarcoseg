import sys
import argparse

from src.preprocessing import preprocess_dicom
from src.segmentation import segment_data
from src.segmentation import segment_tissues

def get_args():
    parser = argparse.ArgumentParser(prog="l3smi", description="segmentation of L3 axial tissues")
    
    sub_parsers = parser.add_subparsers(dest="command", help="select command to run")
    
    preprocess_parser = sub_parsers.add_parser("preprocess",
                                               help="preprocess DICOM files and save as NifTi format",
                                               description="preprocessing options")
    preprocess_parser.add_argument("-i", 
                                   "--input_dir",
                                   type=str, 
                                   help="path to DICOM files",
                                   required=True)
    preprocess_parser.add_argument("-o", 
                                   "--output_dir",
                                   type=str, 
                                   help="path to save preprocessed NifTi files", 
                                   default="./inputs")
    preprocess_parser.add_argument("--anonymize", 
                                   action="store_true", 
                                   help="anonymize DICOM series before saving")
    
    dicom_tags = ("PatientID", "PatientName", "StudyInstanceUID", "StudyDate", "SeriesDescription", "SliceThickness", "PatientSex", "PatientAge", "PatientHeight")
    preprocess_parser.add_argument("--dicom_tags", 
                                   nargs="+", 
                                   help=f"space separated list of additional DICOM tags to extract (default: {dicom_tags})")    
    
    segment_parser = sub_parsers.add_parser("segment", 
                                            help="segment muscle and fat tissue at L3 level in axial viewl",
                                            description="segmentation options")
    segment_parser.add_argument("-i",
                                "--input_dir",
                                type=str,
                                help="path to NifTi data to segment",
                                default="./inputs")
    segment_parser.add_argument("-o",
                                "--output_dir",
                                type=str, 
                                help="path to output directory",
                                default="./outputs")
    segment_parser.add_argument("--slices_num",
                                type=int, 
                                help=("number of slices in superior AND inferior direction from centroid index to extract for tissue segmentation\n"
                                      "example: slices_num=10, extract [centroid_index - 10:centroid_index + 10], segmentation over 20 slices"),
                                default=0)
    segment_parser.add_argument("--add_metrics",
                                nargs="+",
                                help="space separated list of additional metrics to compute", 
                                metavar="metrics")
    segment_parser.add_argument("--save_segmentations", 
                                action="store_true", 
                                help="save segmentation masks")

    return parser.parse_args()

""""
def main():
    in_dir = Path("./outputs/03.nii.gz")
    seg_out = Path("./outputs/03_temp.nii")

    # os.makedirs(out_dir, exist_ok=True)

    target_vertebrae_map = {
        "vertebrae_L1": 31, 
        "vertebrae_L2": 30, 
        "vertebrae_L3": 29, 
        "vertebrae_L4": 28, 
        "vertebrae_L5": 27, 
        "vertebrae_S1": 26
        }
    print(f"targetting {target_vertebrae_map}")


    times = {}
    if not seg_out.exists():
        print("starting total segmentator")
        start = perf_counter()
        totalsegmentator(input=in_dir, output=seg_out, fast=False, ml=True, quiet=False, task="total", roi_subset=list(target_vertebrae_map.keys()), device="gpu")
        duration = perf_counter() - start
        times['spine_seg'] = duration
        print(f"segmentator finished in {duration}s")
    else:
        print("segmentation image exists")

    seg_image: sitk.Image = sitk.ReadImage(seg_out) == target_vertebrae_map['vertebrae_L3']
    # seg_image = sitk.Flip(seg_image, (False, False, True))
    dicom_image: sitk.Image = sitk.ReadImage(in_dir)
    
    label_filter = sitk.LabelShapeStatisticsImageFilter()
    label_filter.Execute(seg_image)
    print(label_filter.GetLabels())
    
    l3_centroid = label_filter.GetCentroid(1)
    l3_centroid_idx = seg_image.TransformPhysicalPointToIndex(l3_centroid)
    print(f"phys {l3_centroid} idx {l3_centroid_idx}")


    seg_image_sagittal = seg_image[l3_centroid_idx[0], ...]
    l3_components_image = sitk.ConnectedComponent(seg_image_sagittal)
    label_filter.Execute(l3_components_image)
    
    l3_body_label = max(label_filter.GetLabels(), key=lambda l: label_filter.GetNumberOfPixels(l))

    l3_body: sitk.Image = l3_components_image == l3_body_label
    
    label_filter.Execute(l3_body)
    l3_body_centroid = label_filter.GetCentroid(1)
    l3_body_centroid_idx = l3_body.TransformPhysicalPointToIndex(l3_body_centroid)

    l3_slice_idxs = (l3_body_centroid_idx[1] - 15, l3_body_centroid_idx[1] + 15)
    l3_slices = dicom_image[..., l3_slice_idxs[0]:l3_slice_idxs[1]]
    l3_slices_path = Path("outputs", "l3_slices_win.nii.gz")
    sitk.WriteImage(l3_slices, l3_slices_path)
    
    target_tissues = {
        "muscle": 1, 
        "sat": 2, 
        "vat": 3, 
        "imat": 4
        }
    
    l3_preds_path = Path("outputs", "l3_slices_preds_win.nii.gz")
    model_dir = Path("models", "muscle_fat_tissue_stanford_0_0_2")
    start = perf_counter()
    f = predict.predict_cases(model=str(model_dir), 
                          list_of_lists=[[l3_slices_path]], 
                          output_filenames=[l3_preds_path],
                          folds="all",
                          save_npz=False,
                          num_threads_preprocessing=8,
                          num_threads_nifti_save=8,
                          segs_from_prev_stage=None,
                          do_tta=False,
                          mixed_precision=True,
                          overwrite_existing=True,
                          all_in_gpu=False,
                          step_size=0.5,
                          checkpoint_name="model_final_checkpoint",
                          segmentation_export_kwargs=None,
                          disable_postprocessing=True
                          )
    duration = perf_counter() - start
    print(f"tissue seg finished in {duration}s")
    times['tissue_seg'] = duration
    print(times, sum(times.values()))
"""

if __name__ == "__main__":
    args = get_args()
    
    if args.command == "preprocess":
        preprocess_dicom(args.input_dir, 
                         args.output_dir, 
                         anonymize=args.anonymize, 
                         dicom_tags=args.dicom_tags)
    elif args.command == "segment":
        segment_data(args.input_dir,
                     args.output_dir,
                     slices_num=args.slices_num,
                     save_segmentations=args.save_segmentations
                     )
        # segment_tissues(args.input_dir,
        #                 args.output_dir,
        #                 vert_center_axial_slices,
        #                 metrics=args.add_metrics,
        #                 slices_range = args.volume_slices,
        #                 save_segmentations=args.save_segmentations,
        #                 )
    else:
        print(f"unknown command '{args.command}'")
        sys.exit(-1)    