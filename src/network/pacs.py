import sys
import os
import subprocess


class PACS:
    def __init__(
        self,
        ip: str,
        port: int,
        aet: str,
        aec: str,
        aem: str = None,
        receiver_port: str = None,
    ):
        self.ip = ip
        self.port = port
        self.aet = aet
        self.aec = aec
        self.aem = aem
        self.receiver_port = receiver_port

    def movescu(self, study_uid: str, output_directory: str):
        result: subprocess.CompletedProcess = None
        command = (
            [
                sys.executable,
                "movescu",
                self.ip,
                self.port,
                "-aec",
                self.aec,
                "-aet",
                self.aet,
                "-aem",
                self.aet,
                "--store",
                "--store-port",
                str(self.receiver_port),
                "-od",
                output_directory,
                "-k",
                "QueryRetrieveLevel=STUDY",
                "-k",
                f"StudyInstanceUID={study_uid}",
            ],
        )

        try:
            result = subprocess.run(command, capture_output=True)
        except subprocess.CalledProcessError as err:
            print(err)

        print(f"finished C-MOVE, status code: {result.returncode}")
