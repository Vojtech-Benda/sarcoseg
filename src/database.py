from labkey.api_wrapper import APIWrapper
from labkey.query import QueryFilter
import requests


API_HANDLER = APIWrapper(
    domain="4lerco.fno.cz", container_path="Sarkopenie/Data", use_ssl=True
)


def query_patients(columns: list[str], patient_id: str):
    if not isinstance(columns, list):
        raise TypeError("positional argument `columns` must be list of strings")

    response = API_HANDLER.query.select_rows(
        schema_name="lists",
        query_name="RDG-CT-Sarko-All",
        columns=",".join(columns),
        max_rows=1,
        filter_array=[
            QueryFilter(
                column="RODNE_CISLO",
                value=patient_id,
                filter_type=QueryFilter.Types.EQUAL,
            )
        ],
    )
    rows = response.get("rows", [])
    if rows:
        return {c: rows[0].get(c, "n/a") for c in columns}
    else:
        return {c: "n/a" for c in columns}


def is_labkey_reachable(verbose=False):
    try:
        response = requests.get(url="https://4lerco.fno.cz", timeout=5)
        if verbose:
            print(f"4lerco.fno.cz reachable: status {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"4lerco.fno.cz unreachable: {e}")
        return False
    return True
