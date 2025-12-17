# idx-flow

**Index-based Spherical Convolutions for HEALPix Grids in PyTorch**

[![PyPI version](https://badge.fury.io/py/idx-flow.svg)](https://badge.fury.io/py/idx-flow)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A PyTorch library for efficient neural network operations on spherical data using HEALPix tessellation. This library implements index-based spherical convolutions that achieve O(N) computational complexity while preserving the equal-area properties essential for atmospheric and geophysical data analysis.

## Citation

This architecture is based on the following paper:

> **Atmospheric Data Compression and Reconstruction Using Spherical GANs**
> Otavio Medeiros Feitosa, Haroldo F. de Campos Velho, Saulo R. Freitas, Juliana Aparecida Anochi, Angel Dominguez Chovert, Cesar M. L. de Oliveira Junior
> DOI: [10.1109/IJCNN64981.2025.11227156](https://doi.org/10.1109/IJCNN64981.2025.11227156)

If you use this library in your research, please cite the paper above.

## Features

- **Efficient O(N) Complexity**: Precomputed neighbor indices enable linear-time convolutions
- **HEALPix Native**: Built for the Hierarchical Equal Area isoLatitude Pixelization scheme
- **PyTorch Integration**: Seamless integration with PyTorch models and training pipelines
- **Flexible Architecture**: Support for encoder-decoder networks, GANs, and custom architectures
- **Multiple Layer Types**: Convolution, transpose convolution, upsampling, MLP, and pooling layers

## Installation

### From PyPI (when published)

```bash
pip install idx-flow
```

### From Source

```bash
git clone https://github.com/otaviomf123/idx-flow.git
cd idx-flow
pip install -e .
```

### Dependencies

- Python >= 3.8
- PyTorch >= 1.9.0
- NumPy >= 1.19.0
- healpy >= 1.15.0
- scikit-learn >= 0.24.0

## Quick Start

### Basic Usage

```python
import torch
from idx_flow import SpatialConv, SpatialTransposeConv, compute_connection_indices

# Compute connection indices for downsampling (nside 64 -> 32)
indices_down, distances_down = compute_connection_indices(
    nside_in=64, nside_out=32, k=4
)

# Compute connection indices for upsampling (nside 32 -> 64)
indices_up, distances_up, weights_up = compute_connection_indices(
    nside_in=32, nside_out=64, k=4, return_weights=True
)

# Create layers
conv = SpatialConv(
    output_points=12 * 32**2,  # 12288 pixels
    connection_indices=indices_down,
    filters=64
)

transpose_conv = SpatialTransposeConv(
    output_points=12 * 64**2,  # 49152 pixels
    connection_indices=indices_up,
    kernel_weights=weights_up,
    filters=32
)

# Forward pass
x = torch.randn(8, 12 * 64**2, 32)  # [batch, points, channels]
encoded = conv(x)                    # [8, 12288, 64]
decoded = transpose_conv(encoded)    # [8, 49152, 32]
```

### Building an Encoder-Decoder Network

```python
import torch
import torch.nn as nn
from idx_flow import (
    SpatialConv,
    SpatialTransposeConv,
    SpatialBatchNorm,
    compute_connection_indices
)

class SphericalAutoencoder(nn.Module):
    """Autoencoder for spherical data on HEALPix grids."""

    def __init__(self, in_channels: int = 5, latent_dim: int = 64):
        super().__init__()

        # Precompute connection indices for each resolution level
        # Encoder: 256 -> 128 -> 64 -> 32
        idx_256_128, _ = compute_connection_indices(256, 128, k=4)
        idx_128_64, _ = compute_connection_indices(128, 64, k=4)
        idx_64_32, _ = compute_connection_indices(64, 32, k=4)

        # Decoder: 32 -> 64 -> 128 -> 256
        idx_32_64, _, w_32_64 = compute_connection_indices(32, 64, k=4, return_weights=True)
        idx_64_128, _, w_64_128 = compute_connection_indices(64, 128, k=4, return_weights=True)
        idx_128_256, _, w_128_256 = compute_connection_indices(128, 256, k=4, return_weights=True)

        # Encoder layers
        self.enc1 = SpatialConv(12*128**2, idx_256_128, filters=32)
        self.enc2 = SpatialConv(12*64**2, idx_128_64, filters=64)
        self.enc3 = SpatialConv(12*32**2, idx_64_32, filters=latent_dim)

        # Decoder layers
        self.dec1 = SpatialTransposeConv(12*64**2, idx_32_64, w_32_64, filters=64)
        self.dec2 = SpatialTransposeConv(12*128**2, idx_64_128, w_64_128, filters=32)
        self.dec3 = SpatialTransposeConv(12*256**2, idx_128_256, w_128_256, filters=in_channels)

        # Batch normalization
        self.bn1 = SpatialBatchNorm(32)
        self.bn2 = SpatialBatchNorm(64)
        self.bn3 = SpatialBatchNorm(latent_dim)

        self.activation = nn.SELU()

    def encode(self, x):
        x = self.activation(self.bn1(self.enc1(x)))
        x = self.activation(self.bn2(self.enc2(x)))
        x = self.activation(self.bn3(self.enc3(x)))
        return x

    def decode(self, z):
        x = self.activation(self.dec1(z))
        x = self.activation(self.dec2(x))
        x = self.dec3(x)
        return x

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

# Example usage
model = SphericalAutoencoder(in_channels=5, latent_dim=64)
x = torch.randn(4, 12*256**2, 5)  # Batch of 4, nside=256, 5 channels
reconstruction = model(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {reconstruction.shape}")
```

## API Reference

### Layers

#### `SpatialConv`

Spatial convolution for downsampling on HEALPix grids.

```python
SpatialConv(
    output_points: int,           # Number of output spatial points
    connection_indices: ndarray,  # [output_points, kernel_size] neighbor indices
    kernel_weights: ndarray = None,  # Optional distance-based weights
    filters: int = 32,            # Number of output channels
    bias: bool = True             # Include bias term
)
```

**Input shape**: `[B, N_in, C_in]`
**Output shape**: `[B, output_points, filters]`

#### `SpatialTransposeConv`

Transpose convolution for upsampling on HEALPix grids.

```python
SpatialTransposeConv(
    output_points: int,           # Number of output spatial points (higher res)
    connection_indices: ndarray,  # [output_points, kernel_size] neighbor indices
    kernel_weights: ndarray = None,  # Distance-based weights (recommended)
    filters: int = 32,            # Number of output channels
    bias: bool = True             # Include bias term
)
```

**Input shape**: `[B, N_in, C_in]` (lower resolution)
**Output shape**: `[B, output_points, filters]` (higher resolution)

#### `SpatialUpsampling`

Non-learnable upsampling using distance-based interpolation.

```python
SpatialUpsampling(
    output_points: int,           # Number of output spatial points
    connection_indices: ndarray,  # [output_points, kernel_size] neighbor indices
    distances: ndarray,           # [output_points, kernel_size] distances
    interpolation: str = "linear",  # "linear", "idw", or "gaussian"
    kernel_radius: float = None   # Radius for interpolation kernel
)
```

**Input shape**: `[B, N_in, C_in]`
**Output shape**: `[B, output_points, C_in]` (channels preserved)

#### `SpatialMLP` (Local Non-Linear Processing)

Unlike `SpatialConv` which applies a **linear kernel** (weighted sum of neighbors), `SpatialMLP` processes each neighborhood through a shared **Multi-Layer Perceptron**. This allows the model to approximate complex **non-linear functions** over the local manifold structure, offering significantly higher expressivity at the cost of increased computational complexity.

**Architectural Difference:**
- `SpatialConv`: `Y = sum_k(W_k * X_k) + b` (linear combination)
- `SpatialMLP`: `Y = MLP([X_1 || X_2 || ... || X_k])` (non-linear transformation)

**Trade-offs:**
| Aspect | SpatialConv | SpatialMLP |
|--------|-------------|------------|
| Expressivity | Linear interactions only | Non-linear interactions |
| Computational Cost | Lower (matrix multiply) | Higher (dense layers) |
| Memory Usage | Lower | Higher |
| Best For | Standard convolutions | Complex local patterns |

This approach is inspired by principles from **Geometric Deep Learning** ([Bronstein et al., 2017](https://doi.org/10.1109/MSP.2017.2693418)) and **Geodesic CNNs** ([Masci et al., 2015](https://doi.org/10.1109/ICCVW.2015.112)).

```python
SpatialMLP(
    output_points: int,                    # Number of output spatial points
    connection_indices: ndarray,           # [output_points, kernel_size]
    hidden_units: tuple = (32, 32, 32),    # Hidden layer dimensions
    activations: tuple = ("gelu", "gelu", "linear"),  # Activation per layer
    dropout: float = 0.0,                  # Dropout probability
    use_batch_norm: bool = False,          # Apply batch normalization
    residual: bool = False,                # Add residual connection
    weight_init: str = "xavier_uniform"    # Weight initialization method
)
```

**Input shape**: `[B, N_in, C_in]`
**Output shape**: `[B, output_points, hidden_units[-1]]`

#### `SpatialPooling`

Pooling operations over local neighborhoods.

```python
SpatialPooling(
    output_points: int,           # Number of output spatial points
    connection_indices: ndarray,  # [output_points, kernel_size]
    pool_type: str = "mean"       # "mean", "max", or "sum"
)
```

**Input shape**: `[B, N_in, C_in]`
**Output shape**: `[B, output_points, C_in]`

#### `SpatialBatchNorm`

Batch normalization for spatial data.

```python
SpatialBatchNorm(
    num_features: int,     # Number of channels
    eps: float = 1e-5,     # Numerical stability constant
    momentum: float = 0.1, # Running statistics momentum
    affine: bool = True    # Learnable affine parameters
)
```

**Input/Output shape**: `[B, N, C]`

### Utility Functions

#### `compute_connection_indices`

Convenience function for computing connection indices and weights.

```python
indices, distances = compute_connection_indices(
    nside_in=64,          # Input resolution
    nside_out=32,         # Output resolution
    k=4                   # Number of neighbors
)

# With weights
indices, distances, weights = compute_connection_indices(
    nside_in=32, nside_out=64, k=4,
    return_weights=True,
    weight_method="inverse_square"
)
```

#### `hp_distance`

Compute geodesic distances and neighbor indices between HEALPix grids.

```python
indices, distances_km = hp_distance(
    nside_in=64,   # Input resolution
    nside_out=32,  # Output resolution
    k=4            # Number of neighbors
)
```

#### `get_weights`

Calculate interpolation weights from distances.

```python
weights = get_weights(
    distances,                    # [N, k] distance array
    method="inverse_square",      # Weighting method
    sigma_factor=0.5,            # For gaussian/exponential
    epsilon=1e-10                # Numerical stability
)
```

Methods: `"inverse_square"`, `"gaussian"`, `"exponential"`, `"tricube"`

#### `get_healpix_resolution_info`

Get information about a HEALPix resolution.

```python
info = get_healpix_resolution_info(nside=256)
# Returns: {
#     'nside': 256,
#     'npix': 786432,
#     'resolution_deg': 0.229,
#     'resolution_km': 25.65,
#     'area_sr': 1.598e-05,
#     'area_km2': 649.17
# }
```

## Mathematical Background

### Index-Based Spherical Convolution

The spatial convolution operation is defined as:

```
Y[b,i,f] = sum_k sum_c N[b,i,k,c] * W[k,c,f] + b[f]
```

Where:
- `Y[b,i,f]`: Output for batch `b`, spatial point `i`, feature channel `f`
- `N[b,i,k,c]`: Neighbor features gathered using precomputed indices
- `W[k,c,f]`: Learnable kernel weights
- `b[f]`: Bias term

### Computational Complexity

| Operation | Traditional | idx-flow |
|-----------|-------------|----------|
| Grid construction | O(N^2) | O(N log N) |
| Neighbor lookup | O(N) | O(1) |
| Convolution | O(N^2) | O(N) |

### HEALPix Grid Properties

- Equal-area pixels: `Area(pixel_i) = 4*pi / N_pix`
- Number of pixels: `N_pix = 12 * nside^2`
- Resolution: Approximately `sqrt(4*pi / N_pix)` radians

## Development

### Setup Development Environment

```bash
git clone https://github.com/otaviomf123/idx-flow.git
cd idx-flow
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest tests/ -v
```

### Code Formatting

```bash
black src/ tests/
isort src/ tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Monan Project**, **CEMPA Project**, **LAMCAD**, and **PGMet**
- CNPq grants (processes 422614/2021-1, and 315349/2023-9)
- National Institute for Space Research (INPE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
