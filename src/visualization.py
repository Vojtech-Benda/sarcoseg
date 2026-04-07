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
    vert_centroid = centroids.vertebre_centroid
    # make body centroid into 3D array only for isotropic resampling
    # left->right direction isn't used anyway
    body_centroid = [vert_centroid[0], *centroids.body_centroid]

    """
    body_centroid: 2D -> 3D array is done only here, instead of directly in get_vertebre_body_centroid().
    Body vertebrae needs to be separated from vertebrae/spine processes as individual islands in sagittal view,
    to accurately compute body centroid.

    """

    ct_volume = sitk.DICOMOrient(ct_volume, "LPS")
    spine_mask = sitk.DICOMOrient(spine_mask, "LPS")

    iso_ct_volume = resample_image_isotropic(ct_volume, sitk.sitkLinear, -1000)
    iso_spine_mask = resample_image_isotropic(spine_mask, sitk.sitkNearestNeighbor, 0)

    iso_vert_centroid = resample_centroid_index(ct_volume, iso_ct_volume, vert_centroid)
    iso_body_centroid = resample_centroid_index(ct_volume, iso_ct_volume, body_centroid)

    sagittal_views = [
        sitk.Cast(
            sitk.IntensityWindowing(
                iso_ct_volume[iso_vert_centroid[0], ...], -100, 900
            ),
            sitk.sitkUInt8,
        ),
        iso_spine_mask[iso_vert_centroid[0], ...],
    ]
    coronal_views = [
        sitk.Cast(
            sitk.IntensityWindowing(
                iso_ct_volume[:, iso_vert_centroid[1], :], -100, 900
            ),
            sitk.sitkUInt8,
        ),
        iso_spine_mask[:, iso_vert_centroid[1], :],
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

    z_index = iso_body_centroid[-1]
    border_margin = 125
    line_width = 1
    line_color = (255, 255, 0)

    x1, x2 = (0 + border_margin, coronal.GetSize()[0] - border_margin)
    y1, y2 = (z_index, z_index + line_width)

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


def resample_image_isotropic(image, interpolator, default_value=0) -> sitk.Image:
    original_spacing = image.GetSpacing()

    if all(spc == original_spacing[0] for spc in original_spacing):
        return sitk.Image(image)

    original_size = image.GetSize()
    min_spacing = min(original_spacing)
    new_spacing = [min_spacing] * image.GetDimension()
    new_size = [
        int(round(osz * ospc / min_spacing))
        for osz, ospc in zip(original_size, original_spacing)
    ]
    new_direction = image.GetDirection()
    new_origin = image.GetOrigin()

    return sitk.Resample(
        image,
        new_size,
        sitk.Transform(),
        interpolator,
        new_origin,
        new_spacing,
        new_direction,
        default_value,
        image.GetPixelID(),
    )


def resample_centroid_index(original_image, iso_image, index):
    phys_point = original_image.TransformIndexToPhysicalPoint(index)
    return iso_image.TransformPhysicalPointToIndex(phys_point)
