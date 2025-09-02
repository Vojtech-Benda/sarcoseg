import numpy as np
from skimage.color import label2rgb
import nibabel as nib
from pathlib import Path
from numpy.typing import NDArray
import imageio


SPINE_COLORS = np.array(
    [
        [0.737, 0.741, 0.133],  # tab:olive
        [0.173, 0.627, 0.173],  # tab:green
        [0.580, 0.403, 0.741],  # tab:purple
        [0.839, 0.153, 0.157],  # tab:red
        [0.122, 0.467, 0.706],  # tab:blue
        [1.000, 0.498, 0.055],  # tab:orange
    ]
)

TISSUE_COLORS = ["khaki", "lightgreen", "dodgerblue", "lightcoral"]

LPI_ORNT = np.array([[0.0, -1.0], [1.0, -1.0], [2.0, -1.0]])  # from RAS to LPI


def overlay_spine_mask(
    spine_vol_path: Path,
    spine_mask_path: Path,
    vert_body_centroid: NDArray,
    output_dir: Path,
    phase: str = None,
):
    # reorient the volume to be in LPI directions for 2D plane
    volumes = [
        nib.as_closest_canonical(nib.load(path))
        for path in (spine_vol_path, spine_mask_path)
    ]
    # extract slice at vertebrae's body centroid and transpose to get the correct orientation on 2D plane
    # slices for each view list: [volume array, mask array]
    coronal = [
        np.squeeze(
            vol.slicer[:, vert_body_centroid[1] : vert_body_centroid[1] + 1, :]
            .as_reoriented(LPI_ORNT)
            .get_fdata(),
            axis=1,
        ).T
        for vol in volumes
    ]
    sagittal = [
        np.squeeze(
            vol.slicer[vert_body_centroid[0] : vert_body_centroid[0] + 1, :, :]
            .as_reoriented(LPI_ORNT)
            .get_fdata(),
            axis=0,
        ).T
        for vol in volumes
    ]

    coronal[0] = normalize_hu(apply_ct_window(coronal[0], width=1000, level=400))
    sagittal[0] = normalize_hu(apply_ct_window(sagittal[0], width=1000, level=400))

    filenames = ("spine_coronal_overlay.png", "spine_sagittal_overlay.png")
    for (image, mask), filename in zip([coronal, sagittal], filenames):
        try:
            overlay = (
                label2rgb(
                    label=mask,
                    image=image,
                    colors=SPINE_COLORS,
                    alpha=0.5,
                    bg_color=None,
                    bg_label=0,
                )
                * 255
            ).astype(np.uint8)

            fullpath = output_dir.joinpath(filename)
            imageio.imsave(fullpath, overlay)

        except RuntimeError as err:
            print(err)


def overlay_tissue_mask(
    tissue_vol_path: Path,
    tissue_mask_path: Path,
    output_dir: Path,
    phase: str = None,
):
    # reorient slices into LPI direction for 2D plane
    image = nib.as_closest_canonical(nib.load(tissue_vol_path)).as_reoriented(LPI_ORNT)
    mask = nib.as_closest_canonical(nib.load(tissue_mask_path)).as_reoriented(LPI_ORNT)

    # tranpose array for the correct axis directions on 2D plane - right hand on left image side, anterior facing upwards
    image, mask = [np.squeeze(img.get_fdata(), axis=-1).T for img in (image, mask)]

    image = normalize_hu(apply_ct_window(image, width=600, level=150))

    try:
        overlay = (
            label2rgb(
                label=mask,
                image=image,
                colors=TISSUE_COLORS,
                alpha=0.8,
                bg_color=None,
                bg_label=0,
            )
            * 255
        ).astype(np.uint8)

        # filename = phase + "_" + "tissue_overlay.png"
        fullpath = output_dir.joinpath("tissue_overlay.png")

        imageio.imsave(fullpath, overlay)
    except RuntimeError as err:
        print(err)


def save_preview_image():
    pass


def apply_ct_window(array, width=400, level=50):
    window_range = (level - width // 2, level + width // 2)
    return np.clip(array, window_range[0], window_range[1])


def normalize_hu(array):
    return (array - array.min()) / (array.max() - array.min())
