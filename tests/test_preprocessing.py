import unittest
from pathlib import Path

from src import preprocessing


class TestPreprocessDicom(unittest.TestCase):
    def test_find_dicom(self):
        path = Path("tests", "input", "dicom")
        files = preprocessing.find_dicoms(path.joinpath("data"))
        no_files = preprocessing.find_dicoms(path.joinpath("empty"))

        self.assertTrue(files)
        self.assertIsNone(no_files)

    def test_dose_report(self):
        filepath = Path("tests/input/dicom/dose", "00000ADD")

        dose_values = preprocessing.extract_dose_values(filepath)

    def test_study_preprocess(self):
        pass


if __name__ == "__main__":
    unittest.main()
