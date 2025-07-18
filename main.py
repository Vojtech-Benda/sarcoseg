import sys
import shutil
import argparse
import os
from pathlib import Path


in_directory = Path("in")

dicom_files = os.listdir(in_directory)
