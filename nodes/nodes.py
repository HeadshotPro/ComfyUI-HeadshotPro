import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import subprocess
import time
import shutil
import os
import hashlib

def download_weights_pget(url, dest, unpack=False):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    if unpack:
        subprocess.check_call(["pget", url, dest], close_fds=False)
    else:
        subprocess.check_call(["pget", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class AlwaysEqualProxy(str):
    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False
    
class DownloadDreamboothCheckpointFromUrl:
    @classmethod
    def INPUT_TYPES(cls):
       return {
            "required": {       
                "url": ("STRING", {"multiline": False, "default": ""}),               
            },
            "optional": {
                "force_download": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = (AlwaysEqualProxy("*"),)
    OUTPUT_NODE = True
    RETURN_NAMES = ("model_path",)
    FUNCTION = "download_file_and_unzip"

    def download_file_and_unzip(self, url, force_download=True):
        dest = "./models/diffusers"
        # Generate a unique ID from the URL
        url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
        file_extension = url.split('.')[-1]  # Extract file extension from URL

        if file_extension == "zip":
            file_path = f"{dest}/{url_hash}.zip"  # Use the hash in the zip file name
        elif file_extension == "tar":
            file_path = f"{dest}/{url_hash}.tar"  # Use the hash in the tar file name
        else:
            raise ValueError("Unsupported file format")
        
        dreambooth_model_dest = f"{dest}/{url_hash}"  # Use the hash in the model folder name

        if force_download or not os.path.exists(dreambooth_model_dest):
            if os.path.exists(dreambooth_model_dest):
                shutil.rmtree(dreambooth_model_dest)
            # Download the file to file_path instead of dest
            if os.path.exists(file_path):
                os.remove(file_path)
            download_weights_pget(url, file_path, unpack=True)  # Assuming you want to unpack
            # Ensure the directory exists
            subprocess.check_call(["mkdir", "-p", dreambooth_model_dest], close_fds=False)
            # Extract the file based on its format
            if file_extension == "zip":
                subprocess.check_call(
                    ["unzip", "-o", file_path, "-d", dreambooth_model_dest], close_fds=False
                )
            elif file_extension == "tar":
                subprocess.check_call(
                    ["tar", "-xvf", file_path, "-C", dreambooth_model_dest], close_fds=False
                )
            print(f"File downloaded and extracted to {dreambooth_model_dest}")
        else:
            print(f"Model already exists at {dreambooth_model_dest}, skipping download.")
        print(os.path.abspath(dreambooth_model_dest))
        return (os.path.abspath(dreambooth_model_dest),)