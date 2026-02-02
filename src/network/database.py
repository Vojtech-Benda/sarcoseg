from labkey.api_wrapper import APIWrapper
from labkey.query import QueryFilter
import requests

from dotenv import dotenv_values

from src import slogger
from src.classes import LabkeyRow

logger = slogger.get_logger(__name__)


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
        verbose=False,
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
        self.verbose = verbose

    def is_labkey_reachable(self):
        hostname = self.server_context.hostname
        try:
            response = requests.get(url=hostname, timeout=5)
            if self.verbose:
                logger.info(f"{hostname} reachable: status {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.critical(f"{hostname} is unreachable")
            logger.critical(f"{e}")
            return False
        return True

    def _select_rows(
        self,
        schema_name: str,
        query_name: str,
        columns: list[str] | str = None,
        max_rows: int = -1,
        filter_dict: dict[str, list[str]] = None,
        sanitize_rows: bool = False,
    ) -> list[dict] | None:
        logger.info("labkey query:")
        logger.info(
            f"domain: {self.server_context.hostname}, schema: {schema_name}, query: {query_name}, columns: {columns}"
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
        logger.info(f"returned rows: {len(rows)}")
        if len(rows) == 0:
            logger.warning("no returned rows")
            return None

        if sanitize_rows:
            return self.sanitize_response_data(rows, columns)
        return rows

    def sanitize_response_data(self, rows: list[dict]):
        return [LabkeyRow.from_labkey_dict(row) for row in rows]

    def _upload_data(
        self, schema_name: str, query_name: str, rows: list, update_rows: bool = False
    ):
        logger.info(f"sending {len(rows)} rows")

        response = None
        if update_rows:
            response = self.query.update_rows(
                schema_name=schema_name, query_name=query_name, rows=rows
            )
        else:
            response = self.query.insert_rows(
                schema_name=schema_name, query_name=query_name, rows=rows
            )

        logger.info(response)


def labkey_from_dotenv(verbose: bool = False) -> LabkeyAPI:
    config = dotenv_values()
    return LabkeyAPI(config["domain"], config["container_path"], verbose=verbose)
