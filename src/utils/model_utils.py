import time

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
    device = next(mode.parameters()).device
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