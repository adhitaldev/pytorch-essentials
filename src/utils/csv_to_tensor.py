"""
Takes in a CSV file and converts it to a tensor.
"""
import torch


def get_tensor_from_csv(csv_file_path: str, delimiter: str = ",", has_header: bool = True) -> torch.tensor:
    """
    Reads a CSV file and converts it to a tensor.
    Args:
        csv_file_path (str): Path to the CSV file.
        delimiter (str): Delimiter used in the CSV file. Default is ','.
    Returns:
        torch.tensor: Tensor representation of the CSV data.
    """
    data = []
    with open(csv_file_path, 'r') as f:
        for line in f:
            if has_header:
                has_header = False
                continue
            row = line.strip().split(delimiter)
            data.append([float(x) for x in row])
    return torch.tensor(data, dtype=torch.float32)