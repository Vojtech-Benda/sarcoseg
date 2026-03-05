from pathlib import Path
import SimpleITK as sitk


SPINE_COLORS = [
    [188, 189, 34],
    [44, 160, 44],
    [148, 103, 189],
    [214, 39, 40],
    [31, 119, 180],
    [255, 127, 14],
]


TISSUE_COLORS_SITK = [
    [51, 153, 255],  # imat
    [255, 80, 80],  # muscle
    [0, 208, 7],  # sat
    [255, 255, 102],  # vat
]


def overlay_spine_mask(
    ct_volume: sitk.Image,
    spine_mask: sitk.Image,
    vert_centroid: list,
    output_dir: Path,
):
    colormap = [channel for color in SPINE_COLORS for channel in color]

    coronal_views = [
        sitk.Cast(
            sitk.IntensityWindowing(ct_volume[vert_centroid[0], ...], -100, 900),
            sitk.sitkUInt8,
        ),
        spine_mask[vert_centroid[0], ...],
    ]
    sagittal_views = [
        sitk.Cast(
            sitk.IntensityWindowing(ct_volume[:, vert_centroid[1], :], -100, 900),
            sitk.sitkUInt8,
        ),
        spine_mask[:, vert_centroid[1], :],
    ]

    coronal = sitk.LabelOverlay(
        coronal_views[0],
        coronal_views[1],
        colormap=colormap,
    )
    sagittal = sitk.LabelOverlay(
        sagittal_views[0],
        sagittal_views[1],
        colormap=colormap,
    )

    sitk.WriteImage(coronal, output_dir.joinpath("spine_coronal_overlay.png"))
    sitk.WriteImage(sagittal, output_dir.joinpath("spine_sagittal_overlay.png"))


def overlay_tissue_mask(
    tissue_volume: sitk.Image,
    tissue_mask: sitk.Image,
    output_dir: Path,
):
    colormap = [channel for color in TISSUE_COLORS_SITK for channel in color]
    tissue_overlay = sitk.LabelOverlay(
        sitk.Cast(sitk.IntensityWindowing(tissue_volume, -135, 215), sitk.sitkUInt8),
        tissue_mask,
        opacity=0.75,
        colormap=colormap,
    )

    sitk.WriteImage(tissue_overlay, output_dir.joinpath("tissue_overlay.png"))
