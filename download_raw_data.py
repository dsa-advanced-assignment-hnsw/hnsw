import subprocess
import requests
import os
import gdown

CACHE = ".cache"
RAW_DATASET_URL = "https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv"
RAW_DATASET_FILE = "train-images-boxable-with-rotation.csv"
DOWNLOAD_TOOL_URL = "https://raw.githubusercontent.com/openimages/dataset/master/downloader.py"
DOWNLOAD_TOOL_FILE = "downloader.py"
subprocess.run(["mkdir", CACHE])
if not os.path.exists(CACHE + "/" + RAW_DATASET_FILE):
  subprocess.run(["wget", "-O", CACHE + "/" + DOWNLOAD_TOOL_FILE, DOWNLOAD_TOOL_URL], check=True)
  print(f"Successfully downloaded and saved the file: {DOWNLOAD_TOOL_FILE}")
else:
  print(f"{DOWNLOAD_TOOL_FILE} already exists")
if not os.path.exists(CACHE + "/" + RAW_DATASET_FILE):
  print(f"Downloading {RAW_DATASET_URL}...")
  try:
      response = requests.get(RAW_DATASET_URL)
      response.raise_for_status()
      with open(CACHE + "/" + RAW_DATASET_FILE, "wb") as f:
          f.write(response.content)
      print(f"Successfully downloaded and saved the file: {RAW_DATASET_FILE}")
  except requests.exceptions.RequestException as e:
      print(f"Error downloading the file: {e}")
else:
  print(f"{RAW_DATASET_FILE} already exists")
if not os.path.isfile(".cache/arxiv-metadata-oai-snapshot.json"):
  gdown.download(id='14QlvPBOCZVLKiqIZ6_7-7pP2lkS8Zd6Z', output='.cache/arxiv-metadata-oai-snapshot.json', quiet=False)