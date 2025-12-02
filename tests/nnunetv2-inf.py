from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from time import perf_counter
import numpy as np
import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

import nibabel as nib


def imshow(arr):
    plt.imshow(arr, vmin=arr.min(), vmax=arr.max())
    plt.show()


def main():
    predictor = nnUNetPredictor()

    predictor.initialize_from_trained_model_folder(
        "models/nnunetv2/ct_muscle_fat_segmentation_weight",
        use_folds=(5,),
        checkpoint_name="checkpoint_best.pth",
    )

    input_files = [["tests/tissue_slices.nii.gz"]]

    start = perf_counter()
    predictor.predict_from_files(
        input_files,
        ["tests/tissue_masks.nii.gz"],
        save_probabilities=False,
        folder_with_segs_from_prev_stage=None,
        num_processes_preprocessing=8,
        num_processes_segmentation_export=8,
    )
    duration = perf_counter() - start
    print(f"finished in {duration}s")
    mask = nib.as_closest_canonical(nib.load("tests/tissue_masks.nii.gz"))

    matplotlib.use("TkAgg")
    plt.figure()
    plt.imshow(mask.get_fdata())
    plt.show()
    # res = []

    # imshow(res[0])
    # plt.imsave("nnunetv2.png", res[0])


if __name__ == "__main__":
    main()
