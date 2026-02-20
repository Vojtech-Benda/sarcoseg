from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import httpx

from src.slogger import get_logger

logger = get_logger(__name__)


MODELS: dict[str, Any] = {
    "totalsegmentator": {"spine": {}},
    "cads": {
        "spine": {
            "task_id": "Dataset552",
            "link": "https://github.com/murong-xu/CADS/releases/download/cads-model_v1.0.0/Dataset552_Totalseg252.zip",
            "dest_folder": "Dataset552_Totalseg252",
        },
        "muscle_fat": {},
    },
    "mazurowski_tissue": {
        "muscle-fat": {
            "link": "https://drive.google.com/uc?export=download&id=1ssy-q1wCqaox4bUwX7xUFOX443saugIX",
            "dest_folder": "mazurowski_muscle_fat",
        }
    },
}


def download_models(working_dir):
    models_dir = Path(working_dir).joinpath("models")

    cads_spine = MODELS["cads"]["spine"]
    url = f"{cads_spine['link']}"

    logger.info(f"downloading model cads-spine from {url}")
    resp = httpx.get(url, follow_redirects=True)
    destination = models_dir.joinpath(cads_spine["dest_folder"])
    with ZipFile(BytesIO(resp.content)) as zip:
        zip.extractall(models_dir)
    logger.info(f"unzip done `{destination}`")

    mazurowski_tissue = MODELS["mazurowski_tissue"]["muscle_fat"]
    url = f"{mazurowski_tissue['link']}"
    logger.info(f"downloading model mazurowski_tissue from {url}")
    resp = httpx.get(url, follow_redirects=True)
    # FIXME: finish some day later
