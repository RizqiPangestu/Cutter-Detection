import json
import os
import requests
from pathlib import Path
from time import time as timer
from requests.auth import HTTPBasicAuth
import cv2

def fetch_url(entry):
    """Download image and save it base on given url and path"""
    save_path, uri = entry
    print(save_path)
    if not os.path.exists(save_path):
        r = requests.get(uri, stream=True)
        if r.status_code == 200:
            with open(save_path, "wb") as file:
                for chunk in r:
                    file.write(chunk)
                    pass
    return save_path


# JSON file exported from labelbox
export_json_path = 'Label/export2.json'

# Output path
images_output_path = os.path.join("dataset", "images")
masks_output_path = os.path.join("dataset", "masks")

# Make path if not exist
Path(images_output_path).mkdir(parents=True, exist_ok=True)
Path(masks_output_path).mkdir(parents=True, exist_ok=True)
Path(os.path.join(masks_output_path, "cutter")).mkdir(parents=True, exist_ok=True)

with open(export_json_path) as f:
    data = json.load(f)

# print(len(data[0]['Label']['objects']))

start = timer()
counter = 0
dataset_size = len(data)

for i in range(0,dataset_size):
    filename = os.path.splitext(data[i]['External ID'])[0]

    if len(data[i]['Label']) > 0: # Check if data labeled
        # Segmentation Label
        segmentation_label_size = len(data[i]['Label']['objects'])
        if segmentation_label_size == 1: # check if label exist
            print("Downloading Image", filename)
            image_url = data[i]["Labeled Data"]
            image_path = os.path.join(images_output_path, filename + ".jpg")
            fetch_url((image_path,image_url))
            counter += 1

            print("Downloading Mask ", filename)
            segmentation_label_value = (data[i]["Label"]["objects"][0]["value"])
            segmentation_label_url = (data[i]["Label"]["objects"][0]["instanceURI"])

            label_path = os.path.join(masks_output_path, segmentation_label_value, filename + ".png")
            fetch_url((label_path, segmentation_label_url))
            counter += 1

            # Generate Cutter Label
            # mask_cutter = cv2.imread(os.path.join(masks_output_path,"cutter",filename + ".png"))
            # cv2.imwrite(os.path.join(masks_output_path, "cutter", filename + ".png"), mask_cutter)

    else:
        print("Missing label ",filename," skipping")

print("Data downloaded: ",counter)
print("Elapsed Time: ",timer()-start,"Seconds")