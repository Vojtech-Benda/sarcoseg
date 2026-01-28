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


if __name__ == "__main__":
    unittest.main()
