import os
import zipfile
from PIL import Image
import json

def zip_images(folder_path, zip_path):
    """
    Zips all image files from `folder_path` into `zip_path`, placing them
    at the root of the resulting ZIP archive.
    """
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    full_path = os.path.join(root, file)
                    zipf.write(full_path, arcname=file)


def convert_png_to_jpg(folder_path):
    """
    Convert all PNG images in 'folder_path' to JPG with a white background
    and remove the original PNG files.
    """
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".png"):
            input_path = os.path.join(folder_path, filename)
            output_path = os.path.join(
                folder_path,
                os.path.splitext(filename)[0] + ".jpg"
            )
            with Image.open(input_path).convert("RGBA") as img:
                white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
                white_bg.paste(img, (0, 0), img)
                rgb_img = white_bg.convert("RGB")
                rgb_img.save(output_path, "JPEG")
            os.remove(input_path)


def update_config():
    # Load existing config
    if os.path.exists("config.json"):
        with open("config.json", "r") as f:
            config = json.load(f)
    else:
        config = {}