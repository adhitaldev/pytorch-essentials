import json
import os
import requests
from PIL import Image
from io import BytesIO

"""
Script to unpack ndjson format
"""

def get_ndjson_with_images(ndjson_file: str, output_folder: str):
    """
    Takes in the ndson file from Ultralytics and then downloads the images 
    and annotations from it.
    """
    train_dir = os.path.join(output_folder, "train")
    val_dir = os.path.join(output_folder, "val")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    with open(ndjson_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            if record.get("type") != "image":
                print(f"Line {line_num}: Skipping non-image record")
                continue

            img_name = record.get("file")
            url = record.get("url")
            labels = record.get("annotations", [])
            response = None
            csplit = record.get("split", None) 
            if csplit == "train":
                csplit = train_dir
            elif csplit == "val":
                csplit = val_dir
            else:
                csplit = output_folder
        
            try:
                # 1. Download and save the image
                response = requests.get(url, timeout=10)
                response.raise_for_status()

                img = Image.open(BytesIO(response.content))
                
                # Convert palette/transparency to RGB if saving as JPEG
                ext = os.path.splitext(img_name)[1].lower()
                if ext in [".jpg", ".jpeg"] and img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                img_path = os.path.join(csplit, img_name)
                img.save(img_path)

                # 2. Save the annotation (.txt for YOLO format)
                base_name = os.path.splitext(img_name)[0]
                label_path = os.path.join(csplit, f"{base_name}.txt")

                with open(label_path, "w", encoding="utf-8") as lf:
                    # Adjust formatting depending on how Ultralytics exports 'label'
                    if isinstance(labels, list):
                        for item in labels:
                            lf.write(f"{item}\n")
                    else:
                        lf.write(f"{labels}\n")

                print(f"Successfully processed {img_name} ({img.size})")

            except requests.exceptions.RequestException as e:
                print(f"Network error for {img_name}: {e}")
            except Exception as e:
                print(f"Failed to process record for {img_name}: {e}")
                if response is not None:
                    print(f"Server response snippet: {response.content[:100]}")
            

if __name__ == "__main__":
    get_ndjson_with_images("/Users/adhital/source/pytorch-essentials/data/dog-pose.ndjson", "/Users/adhital/Documents/ai/keypt_detection/dog_poses")