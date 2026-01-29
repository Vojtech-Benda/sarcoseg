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
        input_path = Path("inputs/1.3.6.1.4.1.36302.1.1.2.67386.4681372")
        output_path = Path("tests/output/1.3.6.1.4.1.36302.1.1.2.67386.4681372")
        study_data = preprocessing.preprocess_dicom_study(input_path, output_path)

        print(study_data)
        self.assertIsNotNone(study_data)

        nifti_files = list(output_path.rglob("*.nii.gz"))
        print(nifti_files)
        self.assertIsNotNone(nifti_files)

        csv_files = list(output_path.rglob("*.csv"))
        print(csv_files)
        self.assertIsNotNone(csv_files)


if __name__ == "__main__":
    unittest.main()
