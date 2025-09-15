from labkey.api_wrapper import APIWrapper
from labkey.query import QueryFilter
import requests

from src.classes import LabkeyData

API_HANDLER = APIWrapper(
    domain="4lerco.fno.cz", container_path="Sarkopenie/Data", use_ssl=True
)


def query_patient_data(
    patient_id: str, query_columns: list[str] = None, max_rows: int = -1
):
    if not isinstance(query_columns, list):
        raise TypeError("positional argument `columns` must be list of strings")

    response = API_HANDLER.query.select_rows(
        schema_name="lists",
        query_name="RDG-CT-Sarko-All",
        columns=",".join(query_columns) if query_columns else None,
        max_rows=max_rows,
        filter_array=[
            QueryFilter(
                column="RODNE_CISLO",
                value=patient_id,
                filter_type=QueryFilter.Types.EQUAL,
            )
        ],
    )
    rows = response.get("rows", [])
    if len(rows) == 0:
        print(
            f"labkey query for {patient_id=} has not matches, queried value possibly not found"
        )
        return None

    queried_data = LabkeyData(
        data={c: rows[0].get(c, "n/a") for c in query_columns},
        query_columns=query_columns,
    )
    return queried_data


def is_labkey_reachable(verbose=False):
    try:
        response = requests.get(url="https://4lerco.fno.cz", timeout=5)
        if verbose:
            print(f"4lerco.fno.cz reachable: status {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"4lerco.fno.cz unreachable: {e}")
        return False
    return True
