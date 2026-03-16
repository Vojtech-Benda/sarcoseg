import logging
from typing import Self

import requests
from labkey.api_wrapper import APIWrapper
from labkey.query import QueryFilter

# from src import slogger
from src.classes import StudyData
from src.io import read_json

log = logging.getLogger("database")


class LabkeyAPI(APIWrapper):
    def __init__(
        self,
        domain,
        container_path,
        context_path=None,
        use_ssl=True,
        verify_ssl=True,
        api_key=None,
        disable_csrf=False,
        allow_redirects=False,
        # debug=False,
    ):
        super().__init__(
            domain,
            container_path,
            context_path,
            use_ssl,
            verify_ssl,
            api_key,
            disable_csrf,
            allow_redirects,
        )
        # self.debug = debug

    def is_labkey_reachable(self):
        hostname = self.server_context.hostname
        reachable = False

        try:
            response = requests.get(url=hostname, timeout=5)
            if response.status_code == 200:
                # log.debug(f"{hostname} reachable: status {response.status_code}")
                reachable = True
            # else:
            # log.debug(f"{hostname} unreachable")
            # return False
            log.debug(f"{response.status_code}")

        except requests.exceptions.RequestException as e:
            # log.critical(f"{hostname} unreachable")
            log.critical(f"{e.response}")
            # reachable = False
        log.info(f"{hostname} reachable: {reachable}")
        return reachable

    def _select_rows(
        self,
        schema_name: str,
        query_name: str,
        columns: list[str] | str | None = None,
        max_rows: int = -1,
        filter_dict: dict[str, list[str]] | None = None,
        sanitize_rows: bool = False,
    ) -> list[StudyData]:
        log.info(
            f"labkey query, schema: {schema_name}, query: {query_name}, columns: {columns}"
        )

        filter_array = None
        if filter_dict:
            filter_array = [
                QueryFilter(
                    column,
                    ";".join(values),
                    QueryFilter.Types.EQUALS_ONE_OF,
                )
                for column, values in filter_dict.items()
            ]

        response = self.query.select_rows(
            schema_name=schema_name,
            query_name=query_name,
            columns=",".join(columns) if isinstance(columns, list) else columns,
            max_rows=max_rows,
            filter_array=filter_array,
        )

        rows = response.get("rows", [])
        log.info(f"returned rows: {len(rows)}")
        if len(rows) == 0:
            log.warning(f"no returned rows from {query_name}")
            return []

        if sanitize_rows:
            return self.sanitize_response_data(rows)
        return rows

    def sanitize_response_data(self, rows: list[dict]) -> list[StudyData]:
        return [StudyData._from_labkey_row(row) for row in rows]

    def _upload_data(
        self, schema_name: str, query_name: str, rows: list, update_rows: bool = False
    ):
        log.info(f"sending {len(rows)} rows")

        response = None
        if update_rows:
            response = self.query.update_rows(
                schema_name=schema_name, query_name=query_name, rows=rows
            )
        else:
            response = self.query.insert_rows(
                schema_name=schema_name, query_name=query_name, rows=rows
            )

        log.info(f"updated {response.get('rowsAffected', 'n/a')}")

    def exclude_finished_studies(self, finished_study_uids: list[str]) -> list[str]:
        """Query Labkey `CTSegmentationData` table with input participants and exclude participants with finished segmentation.
        If the queried table has no data, ie empty response, `input_participants` is returned instead.

        Args:
            input_participants (list[str]): List of participants to query.

        Returns:
            participants (list[str]): List of participants excluding participants existing in the queried table.
        """

        log.info("checking for participants with finished segmnetations")

        rows = self._select_rows(
            schema_name="lists",
            query_name="CTSegmentationData",
            columns=["STUDY_INST_UID"],
            filter_dict={"STUDY_INST_UID": finished_study_uids},
            sanitize_rows=False,
        )

        if not rows:
            log.debug("no rows excluded, returning input")
            return finished_study_uids

        finished_studies = set([row.participant for row in rows])
        log.debug(
            f"excluding {len(finished_studies)} participants due to existing segmentation results"
        )
        return list(set(finished_study_uids).symmetric_difference(finished_studies))

    @classmethod
    def init_from_json(cls, debug: bool = False) -> Self:
        """Initialize Labkey API with configuration values from .env file.

        Args:
            debug (bool, optional): Debug printing for the API. Defaults to False.

        Returns:
            api (LabkeyAPI): LabkeyAPI instance.
        """
        conf = read_json("./src/network/network.json")["labkey"]

        log.debug(f"initializing Labkey API with: {conf}")

        if not all(conf.values()):
            log.error(f"some fields are missing values: {conf}")
            raise ValueError("Unable to initialize LabkeyAPI")

        return cls(conf["domain"], conf["container_path"])
