import time
import torch

def get_model_size_in_mb(model):
    """
    Get the size of the model in MB. This includes both trainable and non-trainable parameters.
    """
    params_size = 0
    # Trainable params
    for param in model.params():
        params_size += param.nelement() * param.element_size()
    # Non-trainable params like buffers, constants, etc
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_in_mb = (params_size + buffer_size) / 1024 ** 2
    return size_in_mb


def get_average_inference_time_ms(model, input_data, num_runs=100):
    """
    Get the average inference time of the model in seconds.
    """
    # Start in evaluation model disabling the dropout layers
    model.eval()
    # Get the model's current device
    device = next(model.parameters()).device
    # Move the input data to the samve device
    input_data = input_data.to(device)
    # Warm up runs - This is crucial to get allow for GPU caching and other optimizations
    # becore calculating the actual number
    with torch.no_grad():
        # Warm up
        for _ in range(10):
            _ = model(input_data)

    # Measure inference time
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_data)
    end_time = time.time()
    avg_inference_time = (end_time - start_time) / num_runs
    return avg_inference_time * 1000  # Convert to milliseconds


def get_model_classes_from_weights_meta(model, weights=None):
    """
    Get the classes of the model from the weights metadata.
    """
    if weights is None:
        raise ValueError("Weights must be provided to get the classes.")
    if not hasattr(weights, 'meta') or 'categories' not in weights.meta:
        raise ValueError("Weights metadata does not contain 'categories'.")
    return weights.meta['categories']

def get_model_classes_from_model(model):
    """
    Get the classes of the model from the model itself.
    """
    if not hasattr(model, 'weights') or model.weights is None:
        raise ValueError("Model does not have weights. Please provide a model with weights.")
    return get_model_classes_from_weights_meta(model, weights=model.weights)


if __name__ == "__main__":
    # Example usage
    from torchvision.models import resnet18, ResNet18_Weights
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    # Create a dummy input tensor with the appropriate shape for ResNet-18
    dummy_input = torch.randn(1, 3, 224, 224)  # Batch size of 1, 3 channels, 224x224 image
    avg_time_ms = get_average_inference_time_ms(model, dummy_input)
    print(f"Average inference time: {avg_time_ms:.2f} ms")

    classes = get_model_classes_from_weights_meta(model, weights=ResNet18_Weights.IMAGENET1K_V1)
    print(f"Model classes: {classes[:5]}...")  # Print first 5 classes for brevity