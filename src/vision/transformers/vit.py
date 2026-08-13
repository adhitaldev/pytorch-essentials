"""
Example ofthe Vision Transformer (ViT) model as descrived here:
https://huggingface.co/docs/transformers/en/model_doc/vit
This examples shows a simple image classification task.
"""
import torch
import os
from transformers import pipeline
from dotenv import load_dotenv
load_dotenv()

def run_vit_classification_example():
    print("--- Running ViT Clasification ----")
    img1 = os.getenv("TEST_IMAGE_1", "")
    pp = pipeline(
            task="image-classification",
            model="google/vit-base-patch16-224",
            dtype=torch.float16,
            device=0)

    result = pp(img1)
    print(result)
    
if __name__== "__main__":
    run_vit_classification_example()