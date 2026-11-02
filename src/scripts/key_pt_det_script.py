"""
Driver script for running the keypoint detector.
"""
import torch
import argparse
import json
from pathlib import Path
import os
from PIL import Image
import torch_vision.keypt_hrnet_w32 as kphrnet
from torch_vision.vision_utils import denormalize_image_tensor, get_pil_image_from_torch_tensor, annotate_key_points_on_image
from utils.model_utils import write_to_csv

image_extensions = ('.jpg', 'jpeg', '.png', 'jpeg')
supported_models = ("hrnet_w32") 
FILE="File"
INFERENCE_OUTPUT_FOLDER="inference_output"
INFERENCE_OUTPUT_FILE="inferences.csv"
INFERENCE_TIME="InferenceTime"

def run_keypt_detection_training():
    kphrnet.run_keydet_training()


def run_keypt_detection_inference(model_path, images_folder, idx_keypoints_json, model_type="hrnet_w32", save=True):
    """
    Runs inference with the given model on all the images in the provided folder.
    Args:
        model_path (str): Path to the trained model checkpoint.
        images_folder (str): Path to the folder containing images for inference.
        idx_keypoints_json (str): Path to the JSON file mapping keypoint indices to names.
        model_type (str): Type of the model. Default is "hrnet_w32".
    """
    print(f"\n-- Running inference with {model_type}\nModel:{model_path}\nImages:{images_folder}\nKeypoints JSON:{idx_keypoints_json}")
    if model_type not in supported_models:
        raise ValueError(f"Unsupported model type: {model_type}. Supported models are: {supported_models}")

    with open(idx_keypoints_json, 'r') as f:
        id_to_keypoints = json.load(f)
        id_to_keypoints = {int(k): v for k, v in id_to_keypoints.items()}  # Convert keys to integers

    print(f"\n-- Loaded keypoint mapping: {id_to_keypoints}")
     # Load the model in evaluation mode
    model = kphrnet.get_trained_model(model_path, num_keypoints=len(id_to_keypoints))

    if not model:
        raise ValueError(f"Model type {model_type} is not supported for inference.")
    
    if save:
        output_folder = os.path.join(images_folder, INFERENCE_OUTPUT_FOLDER)
        output_csv_path = os.path.join(output_folder, INFERENCE_OUTPUT_FILE)
        os.makedirs(output_folder, exist_ok=True)
        print(f"\n-- Inference output will be saved to: {output_folder}")

    model.eval()
    image_paths = [os.path.join(images_folder, file) for file in os.listdir(images_folder) if file.lower().endswith(image_extensions)]
    inferences_times = []
    for file_path in image_paths:
        print(f"\n-- Processing image: {file_path}")
        pil_img = Image.open(file_path).convert("RGB")
        if model_type == "hrnet_w32":
            pil_img, inf_time = kphrnet.infer_on_image(model, pil_img, id_to_keypoints)
            inferences_times.append(inf_time)
        
            if save:
                output_path = os.path.join(output_folder, os.path.basename(file_path))
                pil_img.save(output_path)
                values = {FILE: [file_path], INFERENCE_TIME: [inf_time]}
                write_to_csv(values, output_csv_path)
                print(f"Saved annotated image to: {output_path}")
            else:
                pil_img.show() 
    print(f"Inference times: {inferences_times}")
        

def main():
    parser = argparse.ArgumentParser(description="Run keypoint detector training or inference.")
    parser.add_argument(
        "-train",
        "--train",
        dest="train",
        action="store_true",
        default=True,
        help="Run training (default behavior).",
    )
    parser.add_argument(
        "-infer",
        "--infer",
        dest="infer",
        action="store_true",
        help="Run inference using a model and image path.",
    )
    parser.add_argument("--model_path", type=str, help="Path to the trained model checkpoint.")
    parser.add_argument("--image_path", type=str, help="Path to the input image for inference.")
    parser.add_argument("--idx_kpt_json", type=str, help="Path to the JSON file mapping keypoint indices to names.")
    args = parser.parse_args()

    if args.infer:
        if not args.model_path or not args.image_path or not args.idx_kpt_json:
            parser.error("--infer requires all of --model_path, --image_path, and --idx_kpt_json")
        run_keypt_detection_inference(args.model_path, args.image_path, args.idx_kpt_json)
        return
    elif args.train:
        run_keypt_detection_training()


if __name__ == "__main__":
    main()