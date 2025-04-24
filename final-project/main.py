# 1
# Import necessary modules
import json
import shutil
from download_images import download_google_drive_file, file_ids
from dust3r import send_request
from mast3r import mast3r
import os
from utils import update_config
from preprocess import preprocess
import subprocess
# 2
# input config.json to read configurations

for i in file_ids:
    download_google_drive_file(i)

preprocess()

with open("config.json") as f:
    config = json.load(f)

#---------------------------------------------------------------------------------------------
choice = ""
while choice not in ["d", "m"]:
    config["preprocess_image"] = input("Do you want to preprocess images? Input y/n: ").lower() == 'y'
    choice = input("Choose which model to use ('d' for Dust3r, 'm' for Mast3r): ").lower()
    if choice == "d":
            config["use_dust3r"] = True
            config["use_mast3r"] = False
    elif choice == "m":
            config["use_dust3r"] = False
            config["use_mast3r"] = True
    else:
            print("Invalid input. Please choose 'd' for Dust3r or 'm' for Mast3r.")
    
if config["use_dust3r"] == True:
     dus = send_request("preprocessed.zip")
else:
      mas = mast3r()


#------------------------------------------------------------------------------------------------------------------------   
config["view_output"] = input("Do you want to view the output? Input y/n: ").lower() == 'y'
if config["view_output"] == True:
     try:
          subprocess.run(["python", "-m", "http.server", "8160"])
     except OSError:
            print("Copy and paste this into your shell: 'python -m http.server'")
    
#--------------------------------------------------------    
with open("config.json", "w") as f:
        json.dump(config, f)

print("Configuration updated!")


#-------------------------------------------------------------------------------------------------------------------------
# 3
# implement logic to use different models based on the configured parameters
if config["preprocess_image"]:
    print("We are going to preprocess the image to remove the background")
else:
    print("The images will not be preprocessed.")

if config["use_dust3r"]:
    print("We are going to use dust3r APIs")

if config["use_mast3r"]:
    print("We are going to use mast3r APIs")

