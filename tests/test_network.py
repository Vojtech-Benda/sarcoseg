import unittest
from pathlib import Path

from src.network import pacs, database
from dotenv import dotenv_values


class Pacs(unittest.TestCase):
    api = pacs.pacs_from_dotenv()
    STUDY_INST_UID: str = None

    def test_echoscu(self):
        ret = self.api._echoscu(verbose=True)
        self.assertEqual(ret.returncode, 0)

    def test_movescu(self):
        self.STUDY_INST_UID = dotenv_values().get("test_study_uid")
        validate_study_uid(self.STUDY_INST_UID)

        if not self.STUDY_INST_UID:
            self.fail(
                f"test_movescu requires a valid StudyInstanceUID, given UID is `{self.STUDY_INST_UID}`"
            )

        if "." not in self.STUDY_INST_UID:
            self.fail(f"given StudyInstanceUID is not valid: `{self.STUDY_INST_UID}`")

        download_dir = Path(f"./tests/download/{self.STUDY_INST_UID}")

        if not download_dir.exists() and list(download_dir.rglob("*")) != 0:
            print(
                f"input study directory {self.STUDY_INST_UID} not found, trying to download from PACS instead"
            )
            self.api._movescu(self.STUDY_INST_UID, download_dir)

        files = list(download_dir.rglob("*"))

        self.assertNotEqual(
            len(files), 0, "output download directory empty, no DICOM files downloaded"
        )


class Labkey(unittest.TestCase):
    api = database.labkey_from_dotenv()

    def test_labkey_connection(self):
        self.assertTrue(self.api.is_labkey_reachable(verbose=True))


def validate_study_uid(study_uid: str):
    if "." not in study_uid:
        raise ValueError(
            f"INVALID UID FORMAT: StudyInstanceUID is missing periods `{study_uid}`"
        )
    elif study_uid.islower() or study_uid.isupper():
        raise ValueError(
            f"INVALID UID FORMAT: StudyInstanceUID contains alphabet characters: `{study_uid}`"
        )
    return study_uid


if __name__ == "__main__":
    unittest.main()
