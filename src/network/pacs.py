import sys
import os
import subprocess

from pynetdicom.apps.movescu import movescu


class PACS:
    def __init__(
        self,
        ip: str,
        port: int,
        aet: str,
        aec: str,
        store_port: str,
        aem: str = None,
    ):
        self.ip = ip
        self.port = port
        self.aet = aet
        self.aec = aec
        self.aem = aem if aem else aet
        self.store_port = store_port

    def movescu(self, study_uid: str, output_directory: str):
        result: subprocess.CompletedProcess = None
        print(sys.executable)
        command = [
            sys.executable,
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
            output_directory,
            "-k",
            "QueryRetrieveLevel=STUDY",
            "-k",
            f"StudyInstanceUID={study_uid}",
        ]

        try:
            print("running movescu")
            # result = subprocess.run(command, capture_output=True)
        except subprocess.CalledProcessError as err:
            print(err)

        print(f"finished C-MOVE with status code {result.returncode}")


if __name__ == "__main__":
    pacs = PACS("localhost", 4242, "VOJTPC", "ORTHANC", 2000)
    pacs.movescu(
        "1.2.276.0.7230010.3.1.2.3400784247.11436.1736957593.1146", "./download"
    )
