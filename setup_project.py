import argparse
import os
from zipfile import ZipFile
from pathlib import Path
import huggingface_hub as hf
from totalsegmentator.python_api import download_pretrained_weights


def setup_project(
    input_dir: str = "./inputs",
    output_dir: str = "./outputs",
    download_dir: str = "./downloads",
    remove_model_zip: bool = False,
):
    model_dir = "./models"
    model_name = "muscle_fat_tissue_0_0_2"
    for d in (input_dir, output_dir, model_dir, download_dir):
        if os.path.exists(d):
            print(f"directory `{d}` exists")
            continue
        os.makedirs(d, exist_ok=True)
        print(f"created directory {d}")

    model_path = Path(model_dir, model_name)

    model_repo = "stanfordmimi/multilevel_muscle_adipose_tissue"
    print(f"downloading zipped model {model_name} from huggingface repo {model_repo}")

    model_path.mkdir(exist_ok=True)
    zip_filepath = hf.hf_hub_download(
        repo_id=model_repo, filename="all.zip", local_dir=model_path, repo_type="model"
    )

    if not model_path.joinpath("model_final_checkpoint.model").exists():
        with ZipFile(zip_filepath) as zipf:
            zipf.extractall(model_path)
            print(f"unzipped model to {model_path}")

        if remove_model_zip:
            os.remove(zip_filepath)
            print(f"removed zip file {zip_filepath}")
    else:
        print(f"muscle_adipose_tissue model exists at `{model_path}`")

    """
    TotalSegmentator task ids:
    291 - organs
    292 - vertebrae
    293 - cardiac
    294 - muscles
    295 - ribs
    298 - 6mm model
    """
    task_ids = [291, 292, 293, 294, 295, 298]
    print("downloading TotalSegmentator models")
    for tid in task_ids:
        download_pretrained_weights(tid)


def get_args():
    parser = argparse.ArgumentParser(
        prog="setup_project",
        description="create directories, download models, etc.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        help="path to input directory",
        type=str,
        default="./inputs",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        help="path to output directory",
        type=str,
        default="./outputs",
    )
    parser.add_argument(
        "-d",
        "--download-dir",
        help="path to PACS download directory",
        type=str,
        default="./downloads",
    )
    parser.add_argument(
        "--remove-model-zip",
        action="store_true",
        help="remove downloaded huggingface model ZIP file",
        default=False,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    setup_project(
        args.input_dir,
        args.output_dir,
        args.download_dir,
        remove_model_zip=args.remove_model_zip,
    )
