import os
import requests
from gradio_client import Client, handle_file
import shutil
from pathlib import Path
from download_images import download_google_drive_file, local_paths, file_ids
import zipfile

def mast3r():
    # Step 1: setup directory where images are stored 
    # make it an option in the config. It could be
    # the preprocessed dir or the downloads dir based 
    # on the user choice
    if input == "preprocessed":
        zip_path = 'preprocessed.zip'
        extract_dir = 'preprocessed'

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        local_paths = [os.path.join(extract_dir, fname) for fname in os.listdir(extract_dir)]
    else:
        local_paths = [download_google_drive_file(fid) for fid in file_ids]

    
    # Step 2: Convert local files to gradio-compatible uploads
    filelist = [handle_file(path) for path in local_paths]
    print(filelist)

    # Step 3: Use gradio_client to call the endpoint
    HF_TOKEN = "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"

    if HF_TOKEN is None:
        raise ValueError("ERROR: YOU MUST INPUT YOUR HUGGINGFACE TOKEN")

    # Initialize the client with the Space name
    client = Client("tur-learning/MASt3R", hf_token=f"{HF_TOKEN}")

    # Make the API call
    # You may want to move the most relevant parameters of the model
    # to the config file, to let the user chose without modifying the code
    result = client.predict(
        filelist=filelist,
        min_conf_thr=1.5,
        matching_conf_thr=2,
        as_pointcloud=False,
        cam_size=0.2,
        shared_intrinsics=False,
        api_name="/local_get_reconstructed_scene"
    )

    print("3D model output path:", result)

    print("Copying model to model.glb file")
    shutil.copyfile(result, "model.glb")
