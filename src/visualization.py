from pathlib import Path

import SimpleITK as sitk

from src.classes import Centroids

SPINE_COLORS = [
    [188, 189, 34],
    [44, 160, 44],
    [148, 103, 189],
    [214, 39, 40],
    [31, 119, 180],
    [255, 127, 14],
]


TISSUE_COLORS = [
    [111, 184, 210],  # imat
    [255, 80, 80],  # muscle
    [101, 197, 72],  # sat
    [241, 236, 93],  # vat
]


def overlay_spine_mask(
    ct_volume: sitk.Image,
    spine_mask: sitk.Image,
    centroids: Centroids,
    output_dir: Path,
):

    ct_volume = sitk.DICOMOrient(ct_volume, "LPS")
    spine_mask = sitk.DICOMOrient(spine_mask, "LPS")

    vert_centroid = centroids.vertebre_centroid
    body_centroid = centroids.body_centroid

    sagittal_views = [
        sitk.Cast(
            sitk.IntensityWindowing(ct_volume[vert_centroid[0], ...], -100, 900),
            sitk.sitkUInt8,
        ),
        spine_mask[vert_centroid[0], ...],
    ]
    coronal_views = [
        sitk.Cast(
            sitk.IntensityWindowing(ct_volume[:, vert_centroid[1], :], -100, 900),
            sitk.sitkUInt8,
        ),
        spine_mask[:, vert_centroid[1], :],
    ]

    colormap = [channel for color in SPINE_COLORS for channel in color]
    sagittal = sitk.LabelOverlay(
        sagittal_views[0],
        sagittal_views[1],
        colormap=colormap,
    )
    coronal = sitk.LabelOverlay(
        coronal_views[0],
        coronal_views[1],
        colormap=colormap,
    )

    x1, x2 = (0 + 125, coronal.GetSize()[0] - 125)
    y1, y2 = (body_centroid[-1], body_centroid[-1] + 1)
    line_color = (255, 255, 0)

    # for x in range(x1, x2):
    #     for y in range(y1, y2):
    #         coronal.SetPixel((x, y), line_color)

    for x in range(x1, x2):
        for y in range(y1, y2):
            sagittal.SetPixel((x, y), line_color)

    coronal = sitk.Flip(coronal, [False, True])
    sagittal = sitk.Flip(sagittal, [False, True])

    sitk.WriteImage(coronal, output_dir.joinpath("spine_coronal_overlay.png"))
    sitk.WriteImage(sagittal, output_dir.joinpath("spine_sagittal_overlay.png"))


def overlay_tissue_mask(
    tissue_volume: sitk.Image,
    tissue_mask: sitk.Image,
    output_dir: Path,
):

    colormap = [channel for color in TISSUE_COLORS for channel in color]
    tissue_windowed = sitk.Cast(
        sitk.IntensityWindowing(tissue_volume, -135, 215), sitk.sitkUInt8
    )
    tissue_overlay = sitk.LabelOverlay(
        tissue_windowed,
        tissue_mask,
        opacity=0.75,
        colormap=colormap,
    )

    sitk.WriteImage(tissue_windowed, output_dir.joinpath("tissue_slice.png"))
    sitk.WriteImage(tissue_overlay, output_dir.joinpath("tissue_overlay.png"))
