import unittest
from pathlib import Path

from src import preprocessing
from src.classes import LabkeyRow
import pandas as pd


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
        true_uid = "1.3.6.1.4.1.36302.1.1.2.67388.4692994"
        true_participant = "PAT01"

        input_path = Path("inputs/", true_uid)
        output_path = Path("tests/output/", true_uid)
        row = LabkeyRow(true_participant)

        study_data = preprocessing.preprocess_dicom_study(
            input_path, output_path, labkey_data=row
        )

        self.assertEqual(study_data.participant, true_participant)
        self.assertEqual(study_data.uid, true_uid)

        nifti_files = list(output_path.rglob("*.nii.gz"))
        self.assertIsNotNone(nifti_files)

        csv_files = list(output_path.rglob("*.csv"))
        self.assertIsNotNone(csv_files)

        df = pd.read_csv(csv_files[0])
        test_participant, test_uid = df.participant[0], df.study_inst_uid[0]
        print(
            f"participant: {df.participant[0]}, study inst uid: {df.study_inst_uid[0]}"
        )
        self.assertEqual(test_participant, true_participant)
        self.assertEqual(test_uid, true_uid)


if __name__ == "__main__":
    unittest.main()
