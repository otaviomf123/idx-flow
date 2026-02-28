# idx-flow

**Index-based Spherical Convolutions for HEALPix Grids in PyTorch**

[![PyPI version](https://badge.fury.io/py/idx-flow.svg)](https://badge.fury.io/py/idx-flow)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://readthedocs.org/projects/idx-flow/badge/?version=latest)](https://idx-flow.readthedocs.io/en/latest/)

PyTorch layers for O(N) spherical convolutions on HEALPix grids. Topology (connection indices) is precomputed once; learnable weights are applied at runtime.

**[Documentation](https://idx-flow.readthedocs.io/en/latest/)**

## Citation

Based on the paper:

> **Atmospheric Data Compression and Reconstruction Using Spherical GANs**
> Otavio Medeiros Feitosa, Haroldo F. de Campos Velho, Saulo R. Freitas, Juliana Aparecida Anochi, Angel Dominguez Chovert, Cesar M. L. de Oliveira Junior
> DOI: [10.1109/IJCNN64981.2025.11227156](https://doi.org/10.1109/IJCNN64981.2025.11227156)

If you use this library in your research, please cite the paper above.

## Installation

```bash
pip install idx-flow
```

Upgrade:

```bash
pip install --upgrade idx-flow
```

From source:

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

```python
import torch
from idx_flow import SpatialConv, SpatialTransposeConv, compute_connection_indices

# Downsampling indices (nside 64 -> 32)
indices_down, distances_down = compute_connection_indices(
    nside_in=64, nside_out=32, k=4
)

# Upsampling indices (nside 32 -> 64)
indices_up, distances_up, weights_up = compute_connection_indices(
    nside_in=32, nside_out=64, k=4, return_weights=True
)

conv = SpatialConv(
    output_points=12 * 32**2,
    connection_indices=indices_down,
    filters=64
)

transpose_conv = SpatialTransposeConv(
    output_points=12 * 64**2,
    connection_indices=indices_up,
    kernel_weights=weights_up,
    filters=32
)

x = torch.randn(8, 12 * 64**2, 32)  # [batch, points, channels]
encoded = conv(x)                    # [8, 12288, 64]
decoded = transpose_conv(encoded)    # [8, 49152, 32]
```

### Encoder-Decoder Example

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
    def __init__(self, in_channels: int = 5, latent_dim: int = 64):
        super().__init__()

        # Encoder: 256 -> 128 -> 64 -> 32
        idx_256_128, _ = compute_connection_indices(256, 128, k=4)
        idx_128_64, _ = compute_connection_indices(128, 64, k=4)
        idx_64_32, _ = compute_connection_indices(64, 32, k=4)

        # Decoder: 32 -> 64 -> 128 -> 256
        idx_32_64, _, w_32_64 = compute_connection_indices(32, 64, k=4, return_weights=True)
        idx_64_128, _, w_64_128 = compute_connection_indices(64, 128, k=4, return_weights=True)
        idx_128_256, _, w_128_256 = compute_connection_indices(128, 256, k=4, return_weights=True)

        self.enc1 = SpatialConv(12*128**2, idx_256_128, filters=32)
        self.enc2 = SpatialConv(12*64**2, idx_128_64, filters=64)
        self.enc3 = SpatialConv(12*32**2, idx_64_32, filters=latent_dim)

        self.dec1 = SpatialTransposeConv(12*64**2, idx_32_64, w_32_64, filters=64)
        self.dec2 = SpatialTransposeConv(12*128**2, idx_64_128, w_64_128, filters=32)
        self.dec3 = SpatialTransposeConv(12*256**2, idx_128_256, w_128_256, filters=in_channels)

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
        return self.decode(self.encode(x))
```

## Package Structure

```
idx_flow/
  conv.py           -- SpatialConv, SpatialTransposeConv, SpatialUpsampling
  mlp.py            -- SpatialMLP, GlobalMLP
  norm.py           -- SpatialBatchNorm, SpatialLayerNorm, SpatialInstanceNorm, SpatialGroupNorm
  regularization.py -- SpatialDropout, ChannelDropout
  attention.py      -- SpatialSelfAttention
  vit.py            -- SpatialPatchEmbedding, SpatialTransformerBlock, SpatialViT
  pooling.py        -- SpatialPooling, Squeeze, Unsqueeze
  functional.py     -- get_initializer, get_activation, type aliases
  utils.py          -- hp_distance, get_weights, compute_connection_indices
```

All public names are re-exported from `idx_flow` directly:

```python
from idx_flow import SpatialConv, SpatialViT, SpatialMLP
```

## API Overview

### Convolution (`conv`)

| Layer | Description | Shape |
|---|---|---|
| `SpatialConv` | Convolution via index gathering | `[B, N_in, C]` -> `[B, N_out, F]` |
| `SpatialTransposeConv` | Transpose convolution for upsampling | `[B, N_in, C]` -> `[B, N_out, F]` |
| `SpatialUpsampling` | Distance-weighted interpolation (no params) | `[B, N_in, C]` -> `[B, N_out, C]` |

### MLP (`mlp`)

| Layer | Description | Shape |
|---|---|---|
| `SpatialMLP` | Shared MLP over flattened neighborhoods | `[B, N_in, C]` -> `[B, N_out, H]` |
| `GlobalMLP` | Pointwise MLP (no spatial mixing) | `[B, N, C]` -> `[B, N, H]` |

`SpatialMLP` differs from `SpatialConv` in that it processes the concatenated neighborhood through a full MLP (`Y = MLP([X_1 || ... || X_k])`) rather than a linear kernel (`Y = sum_k W_k X_k + b`). More expressive, but heavier. See [Bronstein et al., 2017](https://doi.org/10.1109/MSP.2017.2693418).

### Normalization (`norm`)

`SpatialBatchNorm`, `SpatialLayerNorm`, `SpatialInstanceNorm`, `SpatialGroupNorm` -- thin wrappers that handle the `[B, N, C]` layout (transpose to `[B, C, N]` internally where needed).

### Regularization (`regularization`)

- `SpatialDropout(p)` -- drops entire spatial locations
- `ChannelDropout(p)` -- drops entire channels

### Attention (`attention`)

`SpatialSelfAttention(embed_dim, num_heads, attn_backend="auto")` -- multi-head self-attention with FlashAttention 2 / SDPA support (PyTorch >= 2.0). O(N^2) in spatial points; use after downsampling on large grids.

### Vision Transformer (`vit`)

| Layer | Description |
|---|---|
| `SpatialPatchEmbedding` | Gather + flatten + linear projection (index-based patch embedding) |
| `SpatialTransformerBlock` | Pre-norm transformer block (MHSA + FFN with residuals) |
| `SpatialViT` | Full ViT: patch embed + positional encoding + N transformer blocks + projection |

### Pooling (`pooling`)

- `SpatialPooling(output_points, indices, pool_type)` -- mean/max/sum over neighborhoods
- `Squeeze(reduction)` -- `[B, N, C]` -> `[B, C]`
- `Unsqueeze(num_points)` -- `[B, C]` -> `[B, N, C]`

### Utilities (`utils`)

- `compute_connection_indices(nside_in, nside_out, k)` -- main entry point for computing topology
- `hp_distance(nside_in, nside_out, k)` -- geodesic neighbor distances
- `get_weights(distances, method)` -- interpolation weights from distances
- `get_healpix_resolution_info(nside)` -- pixel count, resolution in deg/km, area

## Mathematical Background

The convolution gathers k neighbors per output point using precomputed indices, then applies a learnable kernel:

```
Y[b,i,f] = sum_k sum_c X[b, idx[i,k], c] * W[k,c,f] + bias[f]
```

| Operation | Traditional | idx-flow |
|-----------|-------------|----------|
| Grid construction | O(N^2) | O(N log N) |
| Neighbor lookup | O(N) | O(1) |
| Convolution | O(N^2) | O(N) |

## Development

```bash
git clone https://github.com/otaviomf123/idx-flow.git
cd idx-flow
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT -- see [LICENSE](LICENSE).

## Acknowledgments

- Monan Project, CEMPA Project, LAMCAD, PGMet
- CNPq (processes 422614/2021-1 and 315349/2023-9)
- National Institute for Space Research (INPE)

## Contributing

PRs welcome. For larger changes, open an issue first.

1. Fork the repo
2. Create a branch (`git checkout -b feature/my-feature`)
3. Commit and push
4. Open a Pull Request
