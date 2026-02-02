import unittest
from pathlib import Path

from src.classes import MetricsData, Centroids
from src import segmentation


class TestSegmentation(unittest.TestCase):
    def test_study_segment(self):
        test_uid = "1.3.6.1.4.1.36302.1.1.2.67388.4692994"
        path = Path("tests/output/", test_uid)
        segmentation.segment_ct_study(
            input_dir=path,
            output_dir=path,
            save_mask_overlays=True,
            collect_metric_results=True,
        )

    def test_spine_segment(self):
        pass

    def test_tissue_segment(self):
        pass

    def test_tissue_extract(self):
        pass

    def test_postproc_tissue_mask(self):
        pass

    def test_compute_metrics(self):
        pass

    def test_mask_overlays(self):
        pass

    def test_metrics_patient_data(self):
        output_dir = Path("tests/output/1.3.6.1.4.1.36302.1.1.2.67388.4692994")
        all_metrics = [
            MetricsData(
                {
                    "participant": "PAT01",
                    "contrast_phase": "venous",
                },
                area={"area_sat": 200.0},
                mean_hu={"mean_hu_sat": -97.0},
                skelet_muscle_index=28.0,
                duration=35.0,
                centroids=Centroids(
                    vertebre_centroid=[0, 0, 417], body_centroid=[0, 0, 422]
                ),
            ),
            MetricsData(
                {
                    "participant": "PAT01",
                    "contrast_phase": "arterial",
                },
                area={"area_sat": 100.0},
                mean_hu={"mean_hu_sat": -80.0},
                skelet_muscle_index=40.0,
                duration=40.0,
                centroids=Centroids(
                    vertebre_centroid=[0, 0, 500], body_centroid=[0, 0, 414]
                ),
            ),
        ]
        segmentation.write_metric_results(all_metrics, output_dir)


if __name__ == "__main__":
    unittest.main()
