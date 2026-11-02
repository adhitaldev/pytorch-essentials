import torch_vision.obj_det_fastercnn_resnet50_fpn as fcnnres50

def start_object_detection_training():
    fcnnres50.run_obj_det_training()


if __name__ == "__main__":
    start_object_detection_training()  