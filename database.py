import sys
import argparse
from labkey.api_wrapper import APIWrapper


def get_args():
    parser = argparse.ArgumentParser(
        prog="sarcoseg/database",
        description="query labkey database",
    )

    parser.add_argument(
        "-d",
        "--domain",
        default="4lerco.fno.cz",
        help="labkey domain name or ip address",
        type=str,
    )
    parser.add_argument(
        "-c",
        "--container-path",
        default="Sarkopenie/Data",
        help="container path to tables",
        type=str,
    )
    parser.add_argument(
        "-s",
        "--schema-name",
        default="lists",
        choices=("lists"),
        type=str,
    )
    parser.add_argument(
        "-q",
        "--query-name",
        help="name of table to query",
        required=True,
        type=str,
    )

    return parser.parse_args()


def query_column_names(api: APIWrapper, schema_name: str, query_name: str):
    response = api.query.select_rows(
        schema_name=schema_name,
        query_name=query_name,
        max_rows=0,
    )
    return [c["header"] for c in response["columnModel"][1:]]


def format_data(values):


if __name__ == "__main__":
    args = get_args()

    api = APIWrapper(args.domain, args.container_path, use_ssl=True)

    column_names = query_column_names(api, args.schema_name, args.query_name)

    data = 