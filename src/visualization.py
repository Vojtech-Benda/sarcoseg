from pathlib import Path

import nibabel as nib
import numpy as np
import skimage as sk
from nibabel import Nifti1Image
from skimage.color import label2rgb

LPI_ORNT = nib.orientations.axcodes2ornt(("L", "P", "I"))

SPINE_COLORS = [
    [0.580, 0.404, 0.741],
    [0.839, 0.153, 0.157],
    [0.122, 0.467, 0.706],
    [1.0, 0.498, 0.055],
    [0.737, 0.741, 0.133],
    [0.173, 0.627, 0.173],
]

TISSUE_COLORS = [
    [1.0, 0.314, 0.314],
    [0.396, 0.773, 0.282],
    [0.945, 0.925, 0.365],
    [0.435, 0.722, 0.824],
]


def overlay_spine_mask(
    ct_volume: Nifti1Image,
    spine_mask: Nifti1Image,
    vert_centroid: list,
    output_dir: Path,
):

    coronal_views = [
        np.squeeze(
            vol.slicer[:, vert_centroid[1] : vert_centroid[1] + 1, :]
            .as_reoriented(LPI_ORNT)
            .get_fdata(),
            axis=1,
        ).T
        for vol in (ct_volume, spine_mask)
    ]
    sagittal_views = [
        np.squeeze(
            vol.slicer[vert_centroid[0] : vert_centroid[0] + 1, ...]
            .as_reoriented(LPI_ORNT)
            .get_fdata(),
            axis=0,
        ).T
        for vol in (ct_volume, spine_mask)
    ]

    coronal_views[0] = normalize_hu(
        apply_ct_window(coronal_views[0], width=1000, level=400)
    )
    sagittal_views[0] = normalize_hu(
        apply_ct_window(sagittal_views[0], width=1000, level=400)
    )

    coronal_overlay = (
        label2rgb(
            label=coronal_views[1],
            image=coronal_views[0],
            alpha=0.5,
            bg_color=None,
            bg_label=0,
            colors=SPINE_COLORS,
        )
        * 255
    ).astype(np.uint8)
    sagittal_overlay = (
        label2rgb(
            label=sagittal_views[1],
            image=sagittal_views[0],
            alpha=0.5,
            bg_color=None,
            bg_label=0,
            colors=SPINE_COLORS,
        )
        * 255
    ).astype(np.uint8)

    # TODO maybe add later!!
    # if body_centroid:
    #     print(np.array(body_centroid).shape, print(np.array(LPI_ORNT)[1:].shape))
    #     body_centroid = nib.apply_orientation(np.array(body_centroid), LPI_ORNT[1:])
    #     line_coords = sk.draw.line(body_centroid[1], 56, body_centroid[1], 456)
    #     coronal_overlay[line_coords] = np.array(rgb.yellow) * 255
    #     sagittal_overlay[line_coords] = np.array(rgb.yellow) * 255

    sk.io.imsave(output_dir.joinpath("spine_coronal_overlay.png"), coronal_overlay)
    sk.io.imsave(output_dir.joinpath("spine_sagittal_overlay.png"), sagittal_overlay)


def overlay_tissue_mask(
    tissue_volume: Nifti1Image,
    tissue_mask: Nifti1Image,
    output_dir: Path,
):
    tissue_volume = tissue_volume.as_reoriented(LPI_ORNT)
    tissue_mask = tissue_mask.as_reoriented(LPI_ORNT)
    img_array, mask_array = [
        np.squeeze(img.get_fdata(), axis=-1).T for img in (tissue_volume, tissue_mask)
    ]

    img_array = normalize_hu(apply_ct_window(img_array, width=350, level=40))
    overlay = (
        label2rgb(
            label=mask_array,
            image=img_array,
            colors=TISSUE_COLORS,
            alpha=0.75,
            bg_color=None,
            bg_label=0,
        )
        * 255
    ).astype(np.uint8)

    sk.io.imsave(output_dir.joinpath("tissue_overlay.png"), overlay)
    sk.io.imsave(
        output_dir.joinpath("tissue_slice.png"), (img_array * 255).astype(np.uint8)
    )


def apply_ct_window(array, width=400, level=50):
    window_range = (level - width // 2, level + width // 2)
    return np.clip(array, window_range[0], window_range[1])


def normalize_hu(array):
    return (array - array.min()) / (array.max() - array.min())
