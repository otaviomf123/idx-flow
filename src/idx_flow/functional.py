"""
Functional utilities for idx-flow layers.

This module provides type aliases, weight initialization functions, and
activation module constructors used across all spatial layer modules.
"""

from typing import Callable, Literal, Optional

import torch.nn as nn
from torch import Tensor


# =============================================================================
# Type Aliases
# =============================================================================

InterpolationMethod = Literal["linear", "idw", "gaussian"]
PoolingMethod = Literal["mean", "max", "sum"]
InitMethod = Literal["xavier_uniform", "xavier_normal", "kaiming_uniform",
                     "kaiming_normal", "orthogonal", "normal", "uniform", "zeros"]
ActivationType = Literal["relu", "selu", "leaky_relu", "gelu", "elu", "tanh",
                         "sigmoid", "swish", "mish", "linear"]


# =============================================================================
# Initialization Utilities
# =============================================================================


def get_initializer(
    method: InitMethod,
    gain: float = 1.0,
    nonlinearity: str = "leaky_relu",
    mean: float = 0.0,
    std: float = 0.02,
    a: float = 0.0,
    b: float = 1.0,
) -> Callable[[Tensor], Tensor]:
    """
    Get weight initialization function.

    Args:
        method: Initialization method name.
        gain: Gain factor for xavier/orthogonal initialization.
        nonlinearity: Nonlinearity for kaiming initialization.
        mean: Mean for normal initialization.
        std: Standard deviation for normal initialization.
        a: Lower bound for uniform initialization.
        b: Upper bound for uniform initialization.

    Returns:
        Initialization function that takes a tensor and initializes it in-place.

    Raises:
        ValueError: If method is not recognized.
    """
    def init_fn(tensor: Tensor) -> Tensor:
        if method == "xavier_uniform":
            return nn.init.xavier_uniform_(tensor, gain=gain)
        elif method == "xavier_normal":
            return nn.init.xavier_normal_(tensor, gain=gain)
        elif method == "kaiming_uniform":
            return nn.init.kaiming_uniform_(tensor, a=0, mode="fan_in", nonlinearity=nonlinearity)
        elif method == "kaiming_normal":
            return nn.init.kaiming_normal_(tensor, a=0, mode="fan_in", nonlinearity=nonlinearity)
        elif method == "orthogonal":
            return nn.init.orthogonal_(tensor, gain=gain)
        elif method == "normal":
            return nn.init.normal_(tensor, mean=mean, std=std)
        elif method == "uniform":
            return nn.init.uniform_(tensor, a=a, b=b)
        elif method == "zeros":
            return nn.init.zeros_(tensor)
        else:
            raise ValueError(
                f"Unknown initialization method: '{method}'. "
                f"Choose from: xavier_uniform, xavier_normal, kaiming_uniform, "
                f"kaiming_normal, orthogonal, normal, uniform, zeros"
            )
    return init_fn


def get_activation(name: Optional[ActivationType]) -> nn.Module:
    """
    Get activation module by name.

    Args:
        name: Activation function name. If None, returns Identity.

    Returns:
        PyTorch activation module.

    Raises:
        ValueError: If activation name is not recognized.
    """
    activations = {
        "relu": nn.ReLU,
        "selu": nn.SELU,
        "leaky_relu": lambda: nn.LeakyReLU(negative_slope=0.01),
        "gelu": nn.GELU,
        "elu": nn.ELU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "swish": nn.SiLU,  # SiLU is the same as Swish
        "mish": nn.Mish,
        "linear": nn.Identity,
        None: nn.Identity,
    }

    if name not in activations:
        raise ValueError(
            f"Unknown activation: '{name}'. "
            f"Choose from: {list(activations.keys())}"
        )

    act_class = activations[name]
    return act_class() if callable(act_class) else act_class
