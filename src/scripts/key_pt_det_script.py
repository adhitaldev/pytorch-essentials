"""
Driver script for running the keypoint detector.
"""
import torch_vision.keypt_hrnet_w32 as kphrnet

def run_keypt_detection():
    kphrnet.run_keydet_training()


if __name__ == "__main__":
    run_keypt_detection()