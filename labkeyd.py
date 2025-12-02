import argparse
import sys

from src import database


def get_args():
    parser = argparse.ArgumentParser(
        prog="labkeyd",
        description="Data upload to Labkey database at 4lerco.fno.cz.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # parser.add_argument(
    #     "-s",
    #     "--schema",
    #     required=True,
    #     type=str,
    #     help="labkey schema",
    #     choices=("lists", "dataset"),
    #     default="lists",
    # )
    # parser.add_argument(
    #     "-q", "--query-name", required=True, type=str, help="queried table name"
    # )
    # parser.add_argument(
    #     "-d",
    #     "--data-path",
    #     required=True,
    #     type=str,
    #     help="path to .csv table with row data",
    # )
    parser.add_argument(
        "--update-rows", action="store_true", help="update column values at rows"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose printing")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # reached = is_labkey_reachable(args.verbose)
    labkey_api = database.labkey_from_dotenv()
    if not labkey_api.is_labkey_reachable():
        print("labkey is not reachable")

        sys.exit(-1)

    # data = collect_data(args.data_path)

    # send_data(args.schema, args.query_name, rows=data, update_rows=args.update_rows)
