from labkey.api_wrapper import APIWrapper
from labkey.query import QueryFilter
import requests
from pathlib import Path
from typing import Union
import pandas as pd
from src.classes import LabkeyData
from dotenv import dotenv_values
from dataclasses import dataclass


API_HANDLER = APIWrapper(domain="4lerco.fno.cz", container_path="Testy/R", use_ssl=True)


@dataclass
class LabkeyRow:
    patient_id: str
    study_date: str
    study_instance_uid: str
    patient_weight: float = None
    participant: str = None


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

    def is_labkey_reachable(self, verbose=False):
        hostname = self.server_context.hostname
        try:
            response = requests.get(url=hostname, timeout=5)
            if verbose:
                print(f"{hostname} reachable: status {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"{hostname} unreachable: {e}")
            return False
        return True

    def _select_rows(
        self,
        schema_name: str,
        query_name: str,
        columns: list[str] | str = None,
        max_rows: int = -1,
        sanitize_rows: bool = False,
    ) -> list[dict] | None:
        # if isinstance(columns, list):
        #     columns = ",".join(columns)

        print("labkey query:")
        print(
            f"domain: {self.server_context.hostname}, schema: {schema_name}, query: {query_name}, columns: {columns}"
        )

        response = self.query.select_rows(
            schema_name=schema_name,
            query_name=query_name,
            columns=",".join(columns) if isinstance(columns, list) else columns,
            max_rows=max_rows,
        )

        rows = response.get("rows", [])
        print(f"returned rows: {len(rows)}")
        if len(rows) == 0:
            print("no matching rows")
            return None

        if sanitize_rows:
            return self.sanitize_response_data(rows, columns)
        return rows

    def sanitize_response_data(self, rows: list[dict], columns: list[str] | str):
        if isinstance(columns, str):
            columns = columns.split(",")
        return [{col: row[col] for col in row if col in columns} for row in rows]


def labkey_from_dotenv() -> LabkeyAPI:
    config = dotenv_values()
    return LabkeyAPI(config["domain"], config["container_path"])


def send_data(
    schema: str, query_name: str, rows: Union[list, dict], update_rows: bool = False
):
    print(f"sending {len(rows)} rows")
    if update_rows:
        response = API_HANDLER.query.update_rows(
            schema_name=schema, query_name=query_name, rows=rows
        )
    else:
        response = API_HANDLER.query.insert_rows(
            schema_name=schema, query_name=query_name, rows=rows
        )

    if response["rowsAffected"] == 0:
        print(f"labkey {query_name}: no rows affected")


def collect_data(data_path: str) -> Union[None, list]:
    data_path: Path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(1, "data file at path not found", data_path)

    data = pd.read_csv(data_path, header=0, index_col=None)
    return [data.iloc[row_idx].to_dict() for row_idx in data.index]
