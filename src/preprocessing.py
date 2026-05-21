import numpy as np

def min_max_normalize(matrix: np.ndarray) -> np.ndarray:
    min_val = np.min(matrix, axis=0)
    max_val = np.max(matrix, axis=0)
    normalized_matrix = (matrix - min_val) / (max_val - min_val)
    return normalized_matrix

def z_score_normalize(matrix: np.ndarray) -> np.ndarray:
    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0)
    # Adding epsilon to prevent division by zero
    normalized_matrix = (matrix - mean) / (std + 1e-8)
    return normalized_matrix

def preprocess_matrix(matrix: np.ndarray) -> np.ndarray:
    processed_matrix = matrix.astype(np.float32)
    processed_matrix = z_score_normalize(processed_matrix)
    return processed_matrix

def downsample_matrix(matrix: np.ndarray, downsample_rate=250, original_rate=2034) -> np.ndarray:
    factor = original_rate // downsample_rate
    return matrix[:, ::factor]
