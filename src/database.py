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

    def query_patients(
        self,
        schema: str,
        query: str,
        columns: list[str] | str = None,
    ) -> list[LabkeyData]:
        if isinstance(columns, list):
            columns = ",".join(columns)

        print("labkey query:")
        print(
            f"domain: {self.server_context.hostname}, schema: {schema}, query: {query}, columns: {columns}"
        )
        response = self.query.select_rows(
            schema_name=schema, query_name=query, columns=columns
        )

        rows = response.get("rows", [])
        print(f"returned rows: {len(rows)}")
        if rows is None:
            print("no matching rows")
            return []

        queried_data = [normalize_labkey_data(data) for data in rows]
        # for row in rows:
        #     data = LabkeyData(
        #         data={c: row.get(c, "n/a") for c in columns}, query_columns=columns
        #     )
        #     setattr(data, "study_date", normalize_date(data["CAS_VYSETRENI"]))
        #     queried_data.append(data)

        return queried_data


def normalize_labkey_data(data):
    return LabkeyRow(
        patient_id=data["RODNE_CISLO"],
        study_date=normalize_date(data["CAS_VYSETRENI"]),
        study_instance_uid=data["STUDY_INSTANCE_UID"],
        patient_weight=data["VAHA_PAC."],
        participant=data["PARTICIPANT"],
    )


def normalize_date(raw_date_time: str) -> str:
    date = raw_date_time.split(" ")[0]
    return date.replace("-", "")


def construct_api_handler(
    domain: str, container_path: str = None, use_ssl: bool = True
) -> LabkeyAPI:
    return LabkeyAPI(domain=domain, container_path=container_path, use_ssl=use_ssl)


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
