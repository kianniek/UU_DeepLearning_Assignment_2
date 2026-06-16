import numpy as np
import torch

def min_max_normalize(matrix: np.ndarray) -> np.ndarray:
    min_val = np.min(matrix, axis=0)
    max_val = np.max(matrix, axis=0)
    normalized_matrix = (matrix - min_val) / (max_val - min_val)
    return normalized_matrix

def z_score_normalize(matrix: np.ndarray) -> np.ndarray:
    mean = matrix.mean(axis=1, keepdims=True)
    std  = matrix.std(axis=1,  keepdims=True) + 1e-8 # Prevent division by zero
    return (matrix - mean) / std

def preprocess_matrix(matrix: np.ndarray) -> np.ndarray:
    processed_matrix = matrix.astype(np.float32)
    processed_matrix = z_score_normalize(processed_matrix)
    return processed_matrix

def downsample_matrix(matrix: np.ndarray, downsample_rate=250, original_rate=2034) -> np.ndarray:
    factor = original_rate // downsample_rate
    return matrix[:, ::factor]

def add_gaussian_noise(matrix: np.ndarray, std: float = 0.01) -> np.ndarray:
    """Adds random Gaussian noise to the MEG signal."""
    noise = np.random.normal(0, std, matrix.shape).astype(np.float32)
    return matrix + noise

def time_shift(matrix: np.ndarray, shift_range: int = 10) -> np.ndarray:
    """Randomly shifts the signal along the time axis."""
    shift = np.random.randint(-shift_range, shift_range + 1)
    return np.roll(matrix, shift, axis=1)

def scale_amplitude(matrix: np.ndarray, scale_range: tuple = (0.9, 1.1)) -> np.ndarray:
    """Randomly scales the amplitude of the signal."""
    scale = np.random.uniform(scale_range[0], scale_range[1])
    return matrix * scale

def apply_meg_augmentation(matrix: np.ndarray,
                           noise_std: float = 0.01,
                           shift_range: int = 10,
                           scale_range: tuple = (0.9, 1.1),
                           p: float = 0.5) -> np.ndarray:
    """
    Applies a set of MEG-specific augmentations to the input matrix.
    p: Probability of applying augmentation.
    """
    if np.random.random() > p:
        return matrix

    augmented = matrix.copy()
    if np.random.random() > 0.5:
        augmented = add_gaussian_noise(augmented, std=noise_std)
    if np.random.random() > 0.5:
        augmented = time_shift(augmented, shift_range=shift_range)
    if np.random.random() > 0.5:
        augmented = scale_amplitude(augmented, scale_range=scale_range)

    return augmented

def inject_sensor_noise(batch_data, noise_level=0.05):
    """
    Injects Gaussian noise into the MEG sensor matrix.
    Args:
        batch_data (Tensor): Shape (batch_size, channels, time_steps)
        noise_level (float): Scale factor relative to data standard deviation
    """
    if noise_level <= 0:
        return batch_data
    
    # Calculate standard deviation per batch item to scale noise correctly
    std = torch.std(batch_data, dim=(1, 2), keepdim=True)
    noise = torch.randn_like(batch_data) * std * noise_level
    return batch_data + noise
