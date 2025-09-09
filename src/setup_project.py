import os
from zipfile import ZipFile
from pathlib import Path
import huggingface_hub as hf
from totalsegmentator.python_api import download_pretrained_weights


def setup_project(
    input_dir: str = "./inputs",
    output_dir: str = "./outputs",
    model_dir: str = "./models",
    model_name: str = "model_name",
    remove_model_zip: bool = False,
):
    for d in (input_dir, output_dir, model_dir):
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
