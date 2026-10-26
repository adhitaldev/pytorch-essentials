

import os
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from dotenv import load_dotenv
from xml.etree import ElementTree as ET
from torchvision.transforms import v2 as T
from key_pt_dataset import create_keypt_dataset

def prepare_key_pts_data(
    train_images_path,
    train_annotations_path,
    val_images_path,
    val_annotations_path,
    keypt_labels: 'dict',
    batch_size=64,
    train_transforms=None,
    val_transforms=None):
    """
    Prepares data for key point detection based on the given parameters.
    Returns data loaders for training and validation.
    """
    print("\n--------------- PREPARING DATA IN PROGRESS ----------------")
    print(f"Train Images Path: {train_images_path}")
    print(f"Train Annotations Path: {train_annotations_path}")
    print(f"Validate Images Path: {validation_images_path}")
    print(f"Validate Annotations Path: {validation_annotations_path}")
    print(f"Class labels: {class_labels}")
    print(f"Batch size: {batch_size}")

    if not train_transforms:
        train_transforms =  T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=(0.5, 1.4), contrast=(0.8, 1.5), saturation=0.1),
            T.Resize((224, 224))
        ])

    if not validate_transforms:
        validate_transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToDtype(torch.float32, scale=True)
    ])

    train_dataset = create_keypt_dataset(train_images_path, 
                                        train_annotations_path,
                                        keypt_labels, class_labels,
                                        train_transforms)

    val_dataset = create_keypt_dataset(validation_images_path, 
                                        validation_annotations_path,
                                        keypt_labels, class_labels,
                                        validate_transforms) 
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    return train_dataset, val_dataset, train_loader, val_loader


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

def annotate_key_points_on_image(input_img_path: 'str', keypoints: 'list[tuple]', names: 'list[str]',  confidences: 'list[float]', output_path: None, show: True):
    """
    Annotates the given keypoints with names and confidences on the provided image.
    Shows and/or outputs the annotated image if requested.
    """
    if not input_img_path or not keypoints or not names or not confidences:
        print("Incomplete data for annotations. Provide image path, keypoints, names, and confidences")

    if len(keypoints) != len(names) != len(confidences):
        print(f"Keypoints, names, and confidences don't have the same length for annotations")
        return
    image_pil = Image.open(input_img_path).convert('RGB')

    # Redraw and save the image with the annotated points
    try:
        font = ImageFont.truetype("Arial.ttf", size=16)
    except IOError:
        font = ImageFont.load_default()
    x_offset = 10
    y_offset = 10
    radius = 2

    cimg = image_pil.copy()
    draw = ImageDraw.Draw(cimg)
    for keypoint, name, confidence in zip(keypoints, names, confidences):
        # Draw a ellipse around the x and y for visibility
        x, y = keypoint[0], keypoint[1]
        bounding_box = [x - radius, y - radius, x + radius, y + radius]
        draw.ellipse(bounding_box, fill="red")
        # Draw the text
        text_coords = (x + x_offset, y - y_offset)
        txt_to_draw = name + "_" + str(confidence)
        draw.text(text_coords, name, fill="black", font=font)

    # save
    if output_path:
        img_out_path = os.path.join(output_folder, img)
        cimg.save(img_out_path)
    if show:
        cimg.show()



if __name__ == "__main__":
    env_file = ".env.keypts_detection"
    if os.path.exists(env_file):
      load_dotenv(env_file)
    else:
      load_dotenv()
    
    images_folder = os.getenv("KEYPT_DETECTION_TRAIN_IMAGES_PATH", "")
    annotations_folder = os.getenv("KEYPT_DETECTION_TRAIN_ANNOTATIONS_PATH", "")
    output_folder = os.getenv("KEYPT_DETECTION_OUTPUT_PATH", "")
    annotate_keypoints_from_xml_images(images_folder, annotations_folder, output_folder)