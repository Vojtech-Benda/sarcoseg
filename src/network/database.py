from labkey.api_wrapper import APIWrapper
from labkey.query import QueryFilter
import requests
from pathlib import Path
from typing import Union
import pandas as pd

# from src.classes import LabkeyData
from dotenv import dotenv_values
from dataclasses import dataclass


API_HANDLER = APIWrapper(domain="4lerco.fno.cz", container_path="Testy/R", use_ssl=True)


@dataclass
class LabkeyRow:
    patient_id: str
    study_date: str
    participant: str
    study_instance_uid: str
    pacs_number: str = None
    patient_weight: float = None


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
        filter_dict: dict[str, list[str]] = None,
        sanitize_rows: bool = False,
    ) -> list[dict] | None:
        print("labkey query:")
        print(
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

        ret: list[LabkeyRow] = []
        for row in rows:
            patient_data = LabkeyRow(
                patient_id=row.get("RODNE_CISLO"),
                study_date=row.get("CAS_VYSETRENI").split(" ")[
                    0
                ],  # take date, discard time ["date", "time"]
                participant=row.get("PARTICIPANT"),
                study_instance_uid=row.get("STUDY_INSTANCE_UID"),
                pacs_number=row.get("PACS_CISLO"),
                patient_weight=row.get("VYSKA_PAC."),
            )

            ret.append(patient_data)
        # return [{col: row[col] for col in row if col in columns} for row in rows]
        return ret

    def _upload_data(
        self, schema_name: str, query_name: str, rows: list, update_rows: bool = False
    ):
        print(f"sending {len(rows)} rows")

        response = None
        if update_rows:
            response = self.query.update_rows(
                schema_name=schema_name, query_name=query_name, rows=rows
            )
        else:
            response = self.query.insert_rows(
                schema_name=schema_name, query_name=query_name, rows=rows
            )

        print(response)


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
