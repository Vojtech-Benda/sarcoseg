import sys
import os
import subprocess
from dotenv import dotenv_values
from pathlib import Path

from pynetdicom.apps.echoscu import echoscu
from pynetdicom.apps.movescu import movescu


class PacsAPI:
    def __init__(
        self,
        ip: str,
        port: int,
        aet: str,
        aec: str,
        store_port: int,
        aem: str = None,
    ):
        self.ip = ip
        self.port = port
        self.aet = aet
        self.aec = aec
        self.aem = aem if aem else aet
        self.store_port = store_port

    def _movescu(self, study_inst_uid: str, download_directory: str):
        if (
            os.path.exists(download_directory)
            and len(os.listdir(download_directory)) != 0
        ):
            print(
                f"skipping existing directory with DICOM files for study uid {study_inst_uid}"
            )
            return 0

        os.makedirs(download_directory, exist_ok=True)

        args = [
            movescu.__file__,
            self.ip,
            str(self.port),
            "-aec",
            self.aec,
            "-aet",
            self.aet,
            "-aem",
            self.aem,
            "--store",
            "--store-port",
            str(self.store_port),
            "-od",
            download_directory,
            "-k",
            "QueryRetrieveLevel=STUDY",
            "-k",
            f"StudyInstanceUID={study_inst_uid}",
        ]

        print(f"running C-MOVE for StudyInstanceUID: {study_inst_uid}")
        ret = subprocess.run(args, capture_output=True, text=True)

        if len(os.listdir(download_directory)) == 0:
            print(
                f"C-MOVE failed downloading data for StudyInstanceUID: {study_inst_uid}"
            )
            return -1

        print("finished C-MOVE")
        return 0

    def _echoscu(self, verbose: bool = False):
        args = [
            sys.executable,
            echoscu.__file__,
            self.ip,
            str(self.port),
            "-aec",
            self.aec,
            "-aet",
            self.aet,
        ]

        if verbose:
            args.append("-v")

        ret = subprocess.run(args, capture_output=True, text=True)
        print(ret.returncode)

        if verbose:
            print(ret.stdout)


def pacs_from_dotenv(verbose: bool = False):
    config = dotenv_values()

    if verbose:
        print(config)

    return PacsAPI(
        config["pacs_ip"],
        int(config["pacs_port"]),
        config["aet"],
        config["aec"],
        int(config["store_port"]),
    )


if __name__ == "__main__":
    pacs = PacsAPI("localhost", 4242, "VOJTPC", "ORTHANC", 2000)
    pacs._echoscu()
