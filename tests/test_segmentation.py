import unittest
from pathlib import Path

from src import segmentation
from src import utils


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
        pass


if __name__ == "__main__":
    unittest.main()
