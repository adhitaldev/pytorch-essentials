

import os
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from dotenv import load_dotenv
from xml.etree import ElementTree as ET
from torchvision.transforms import v2 as T
from utils.model_utils import read_from_csv
from utils.data_plotter import plot_x_y, plot_multiX_y, plot_multiY_x

ENV_FILE_NAME = ".env.keypts_detection"
LR="Lr"
EPOCH="Epoch"
STEP="Step"
LOSS="Loss"
ARTIFACTS_FOLDER="artifacts"
MEAN_PIXEL_ERROR="Mean_PX_Error"
MEDIAN_PIXEL_ERROR="Median_PX_Error"
TRAIN_LOSS="AvTrainLoss"
VALIDATION_LOSS="AvValidateLoss"
PRECISION="Precision"
RECALL="Recall"
F1_SCORE="F1_Score"

# Artifacts definitons
ANALYTICS_FILE="analytics.csv"
STEP_VS_LOSS_FILE="step_vs_loss.jpg"
EPOCH_EVAL_RESULTS="eval_results.csv"
EPOCH_PIXEL_ERROR_PLOT="pixel_errors.jpg"
EPOCH_TRAIL_VAL_LOSS="train_vs_validate_loss.jpg"
EPOCH_PRF1_PLOT ="precision_recall_f1.jpg"

def load_keypts_env():
    """Load .env.keypts_detection from the repository root, independent of cwd."""
    env_file = Path(__file__).resolve().parents[2] / ENV_FILE_NAME
    if not env_file.is_file():
        raise FileNotFoundError(f"Keypoint detection env file not found: {env_file}")
    if not load_dotenv(env_file):
        raise RuntimeError(f"Unable to load env file: {env_file}")
    print(f"Loaded environment from {env_file}")
    return env_file

def create_keypts_output_infrastructure(output_path, model_name):
    """
    Creates the necessary folder strucutures to output the model, artifacts etc.
    """
    folder_name = datetime.now().strftime(f"%Y%m%d_%H%M") + "_" + model_name
    final_path = os.path.join(output_path, folder_name)
    model_path = os.path.join(final_path, model_name + ".pt")
    artifacts_path = os.path.join(final_path, ARTIFACTS_FOLDER)
    analytics_path = os.path.join(final_path, ANALYTICS_FILE)
    epoch_results_path = os.path.join(final_path, EPOCH_EVAL_RESULTS)
    if not os.path.exists(artifacts_path):
        os.makedirs(artifacts_path)
    return model_path, artifacts_path, analytics_path, epoch_results_path

def generate_final_analytics(analytics_path, epoch_results_path, artifacts_path):
    """
    Reads from the analytics files and then generates the artifacts
    """
    print(f"Preparing analytics ...")
    # Getting and plotting data from the analytics file
    data = read_from_csv(analytics_path)
    print(f"Reading from analytics file: {analytics_path}")
    steps = data[STEP].tolist()
    losses = data[LOSS].tolist()
    save_path=os.path.join(artifacts_path, STEP_VS_LOSS_FILE)
    plot_x_y(steps, losses, x_label="steps", y_label="loss", title="Step vs Loss", save_path=save_path)

    # Getting and plotting data from the pixel error file
    print(f"Reading from pixel error file: {epoch_results_path}")
    mdata = read_from_csv(epoch_results_path)
    epochs = mdata[EPOCH].tolist()
    train_loss = mdata[TRAIN_LOSS].tolist()
    val_loss = mdata[VALIDATION_LOSS].tolist()
    mean_error = mdata[MEAN_PIXEL_ERROR].tolist()
    median_error = mdata[MEDIAN_PIXEL_ERROR].tolist()
    precision = mdata[PRECISION].tolist()
    recall = mdata[RECALL].tolist()
    f1score = mdata[F1_SCORE].tolist()

    # Mean and median pixel errors on epochs
    y_values = [mean_error, median_error]
    y_names = ["Mean Px Error", "Median Px Error"]
    y_colors = ["green", "orange"]
    save_path = os.path.join(artifacts_path, EPOCH_PIXEL_ERROR_PLOT)
    plot_multiY_x(y_values=y_values, y_names=y_names, y_colors=y_colors, x_values=epochs, x_title="Epochs", title="Keypoints Pixel Errors", save_path=save_path)

    # Training vs Validation loss on epochs
    y_values = [train_loss, val_loss]
    y_names = [TRAIN_LOSS, VALIDATION_LOSS]
    y_colors = ["blue", "red"]
    save_path = os.path.join(artifacts_path, EPOCH_TRAIL_VAL_LOSS)
    plot_multiY_x(y_values=y_values, y_names=y_names, y_colors=y_colors, x_values=epochs, x_title="Epochs", title="Training vs Validation Loss", save_path=save_path)

    # Precision, recall and f1 score
    y_values = [precision, recall, f1score]
    y_names = [PRECISION, RECALL, F1_SCORE]
    y_colors = ["blue", "green", "grey"]
    save_path = os.path.join(artifacts_path, EPOCH_PRF1_PLOT)
    plot_multiY_x(y_values=y_values, y_names=y_names, y_colors=y_colors, x_values=epochs, x_title="Epochs", title="Precision Recall F1Score", save_path=save_path)


def annotate_keypoints_from_xml_images(images_folder, annotations_folder, output_folder, split_extension=True):
    """
    Draws the keypoints for each image in the images_folder from respective .xml annotation file
    in the annotations_folder. It creates a new image and outputs it to the output_folder.
   
    Split extension is true if image name and annotations look like this:
    image1.jpg and image1.xml
    It's not to be set to True if image name and annotations look like this:
    image1.jpg and image1.jpg.xml
    
    The format of each annotation looks like this:
    <annotation>
        <filename>file_name.jpg</filename>
        <size>
            <width>640</width>
            <height>480</height>
            <depth>3</depth>
        </size>
        <object>
            <name>class_name</name>
            <point>
            <x>400.042672</x>
            <y>91.249030</y>
            </point>
        </object>
    </annotation>
    File name are as follows: image1.jpg & image1.xml
    """
    print(f"\n-- Creating keypoint annotations ----")
    print(f"Image folder: {images_folder}")
    print(f"Annotations folder: {annotations_folder}")
    img_extensions =('.jpg', 'jpeg', '.png', 'jpeg')
    anno_extensions = '.xml'

    images = set()
    annotations = set()
    for file in os.listdir(images_folder):
        if file.lower().endswith(img_extensions):
            images.add(file)
    
    for file in os.listdir(annotations_folder):
        if file.lower().endswith(anno_extensions):
            annotations.add(file)
    
    # Check if there are any missing annotations
    missing_annotations = []
    for img in images:
        if not split_extension:
            img_with_xml = img + anno_extensions
        else:
            img_path = Path(img)
            fname = img_path.stem
            img_with_xml = fname + anno_extensions
        if img_with_xml not in annotations:
            missing_annotations.append(img_with_xml)
    
    # Output folder 
    if not os.path.exists(output_folder):
        print(f"Output folder not available. Creating ...{output_folder}")
        os.makedirs(output_folder)

    images = sorted(images)
    annotations = sorted(annotations)

    print(f"Total Images: {len(images)}")
    print(f"Total Annotations: {len(annotations)}")
    print(f"Missing Annotations for :{missing_annotations}")

    # Extract and draw
    for img, anno in zip(images, annotations):
        img_p = os.path.join(images_folder, img)
        anno_p = os.path.join(annotations_folder, anno)

        image_pil= Image.open(img_p).convert('RGB')
        width_height = image_pil.size

        # Extract teh annotations
        tree = ET.parse(anno_p)
        root = tree.getroot()
        names, points = [], []
        for obj in root.findall('object'):
            name = obj.find('name').text
            pts = obj.find('point') if name else None
            x = float(pts.find('x').text) if pts else None
            y = float(pts.find('y').text) if pts else None
            if not name or not pts:
                continue
            names.append(name)
            points.append((x, y))

        # Redraw and save the image with the annotated names/points
        try:
            font = ImageFont.truetype("Arial.ttf", size=16)
        except IOError:
            font = ImageFont.load_default()
        x_offset = 10
        y_offset = 10
        radius = 2
        for name, pt in zip(names, points):
            cimg = image_pil.copy()
            draw = ImageDraw.Draw(cimg)
            # Draw a ellipse around the x and y for visibility
            x, y = pt[0], pt[1]
            bounding_box = [x - radius, y - radius, x + radius, y + radius]
            draw.ellipse(bounding_box, fill="red")
            # Draw the text
            text_coords = (x + x_offset, y - y_offset)
            draw.text(text_coords, name, fill="black", font=font)
            # save
            img_out_path = os.path.join(output_folder, img)
            cimg.save(img_out_path)
    print(f"{len(images) - len(missing_annotations)} images annotated and saved to {output_folder}")

