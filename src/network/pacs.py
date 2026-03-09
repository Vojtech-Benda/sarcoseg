import os
import subprocess
import sys
from pathlib import Path
from typing import Self

from pydicom import dcmread
from pynetdicom.apps.echoscu import echoscu
from pynetdicom.apps.movescu import movescu

from src import slogger
from src.io import read_json

logger = slogger.get_logger(__name__)

WRONG_IMAGE_TYPES = ["DERIVED", "SECONDARY", "OTHER", "LOCALIZER"]


class PacsAPI:
    def __init__(
        self,
        ip: str,
        port: int,
        aet: str,
        aec: str,
        store_port: int,
        aem: str | None = None,
    ):
        self.ip = ip
        self.port = port
        self.aet = aet
        self.aec = aec
        self.aem = aem if aem else aet
        self.store_port = store_port

    def _movescu(self, study_inst_uid: str, download_directory: str | Path):
        os.makedirs(download_directory, exist_ok=True)

        # args_findscu = [
        #     "./dcmtk/findscu",
        #     self.ip,
        #     str(self.port),
        #     "-aet",
        #     self.aet,
        #     "-aec",
        #     self.aec,
        #     "-S",
        #     "-X",  # store responses
        #     "-k",
        #     "QueryRetrieveLevel=SERIES",
        #     "-k",
        #     f"StudyInstanceUID={study_inst_uid}",
        #     "-od",
        #     str(download_directory),
        # ]
        # result = subprocess.run(args_findscu, capture_output=True, text=True)
        # print(result.returncode)
        # if result.returncode == -1
        #     return result.returncode

        # # TODO: filter out response files
        # response_files = list(Path(download_directory).rglob("rsp*.dcm"))

        # query_files = []
        # for res_file in response_files:
        #     ds = dcmread(res_file)

        #     series_des = ds.get("SeriesDescription", "")
        #     if series_des and "dose report" in series_des.lower():
        #         query_files.append(res_file)

        #     image_type = ds.get("ImageType", [])
        #     if image_type and any(img_type in WRONG_IMAGE_TYPES for img_type in image_type):
        #         continue

        #     query_files.append(res_file)

        # # clean the download directory
        # [os.remove(f) for f in response_files]

        args = [
            "./dcmtk/movescu",
            self.ip,
            str(self.port),
            "-aec",
            self.aec,
            "-aet",
            self.aet,
            "--port",
            str(self.store_port),
            "-od",
            str(download_directory),
            "-S",
            "-k",
            "QueryRetrieveLevel=STUDY",
            "-k",
            f"StudyInstanceUID={study_inst_uid}",
            # TODO: add response files here!!!
        ]

        logger.info(f"running C-MOVE for StudyInstanceUID: {study_inst_uid}")
        try:
            result = subprocess.run(args, capture_output=True, text=True)
            # movescu.main(args)
        except SystemExit:
            logger.info("movescu finished and tried to exit; continuing execution")

        if len(os.listdir(download_directory)) == 0:
            os.rmdir(download_directory)
            logger.info(
                f"C-MOVE failed downloading data for StudyInstanceUID: {study_inst_uid}"
            )
            return -1

        logger.info("finished C-MOVE")
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
        logger.info(f"ECHOSCU return code: {ret.returncode}")

        if verbose:
            if ret.stdout:
                logger.info(f"ECHOSCU stdout: {ret.stdout}")
            if ret.stderr:
                logger.info(f"ECHOSCU stderr: {ret.stderr}")
        return ret

    @classmethod
    def init_from_json(cls, verbose: bool = False) -> Self:
        conf = read_json("./src/network/network.json")["pacs"]

        if verbose:
            logger.info(f"initializing PACS API with: {conf}")

        if not all(conf.values()):
            logger.error(f"some fields are missing values: {conf}")
            raise ValueError("Unable to initialize PACS API")

        return cls(
            conf["ip"],
            int(conf["port"]),
            conf["aet"],
            conf["aec"],
            int(conf["store_port"]),
        )


if __name__ == "__main__":
    pacs = PacsAPI.init_from_json()
    pacs._echoscu()
