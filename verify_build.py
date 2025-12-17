#!/usr/bin/env python3
"""
verify_build.py - Runtime Verification Script for idx-flow Package

This script performs dynamic analysis to verify the integrity and correctness
of the idx-flow package before deployment.

Checks performed:
    1. Import Verification - Ensure all core layers can be imported
    2. Shape Verification - Forward pass produces correct output shapes
    3. Gradient Verification - Backward pass allows gradient flow
    4. New Features Verification - Test initialization options and new layers

Architecture based on the paper:
    Atmospheric Data Compression and Reconstruction Using Spherical GANS.
    DOI: 10.1109/IJCNN64981.2025.11227156

Author: QA Verification Script
"""

import sys
from typing import Callable, Dict, List, Tuple

import numpy as np


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_result(test_name: str, passed: bool, details: str = "") -> None:
    """Print test result with PASS/FAIL status."""
    status = "\033[92mPASS\033[0m" if passed else "\033[91mFAIL\033[0m"
    print(f"  [{status}] {test_name}")
    if details:
        print(f"         {details}")


def run_test(test_name: str, test_fn: Callable[[], Tuple[bool, str]]) -> bool:
    """Run a single test and handle exceptions."""
    try:
        passed, details = test_fn()
        print_result(test_name, passed, details)
        return passed
    except Exception as e:
        print_result(test_name, False, f"Exception: {e}")
        return False


# =============================================================================
# Test Utilities
# =============================================================================


def create_dummy_indices(
    output_points: int, input_points: int, k: int
) -> np.ndarray:
    """Create dummy connection indices for testing."""
    rng = np.random.default_rng(42)
    indices = rng.integers(0, input_points, size=(output_points, k))
    return indices.astype(np.int64)


def create_dummy_distances(output_points: int, k: int) -> np.ndarray:
    """Create dummy distances for testing."""
    rng = np.random.default_rng(42)
    distances = rng.uniform(0.1, 1.0, size=(output_points, k))
    return distances.astype(np.float64)


# =============================================================================
# Test 1: Import Verification
# =============================================================================


def test_import_core_layers() -> Tuple[bool, str]:
    """Test import of core spatial layers."""
    from idx_flow import SpatialConv, SpatialTransposeConv, SpatialUpsampling, SpatialPooling
    return True, "Core layers imported successfully"


def test_import_mlp_layers() -> Tuple[bool, str]:
    """Test import of MLP layers."""
    from idx_flow import SpatialMLP, GlobalMLP
    return True, "MLP layers imported successfully"


def test_import_normalization_layers() -> Tuple[bool, str]:
    """Test import of normalization layers."""
    from idx_flow import SpatialBatchNorm, SpatialLayerNorm, SpatialInstanceNorm, SpatialGroupNorm
    return True, "Normalization layers imported successfully"


def test_import_regularization_layers() -> Tuple[bool, str]:
    """Test import of regularization layers."""
    from idx_flow import SpatialDropout, ChannelDropout
    return True, "Regularization layers imported successfully"


def test_import_attention_layers() -> Tuple[bool, str]:
    """Test import of attention layers."""
    from idx_flow import SpatialSelfAttention
    return True, "Attention layers imported successfully"


def test_import_utility_layers() -> Tuple[bool, str]:
    """Test import of utility layers."""
    from idx_flow import Squeeze, Unsqueeze
    return True, "Utility layers imported successfully"


def test_import_utilities() -> Tuple[bool, str]:
    """Test import of utility functions."""
    from idx_flow import get_initializer, get_activation, compute_connection_indices
    return True, "Utility functions imported successfully"


# =============================================================================
# Test 2: Shape Verification
# =============================================================================


def test_spatial_conv_shape() -> Tuple[bool, str]:
    """Test SpatialConv output shape."""
    import torch
    from idx_flow import SpatialConv

    batch_size, input_points, input_channels = 2, 192, 16
    output_points, kernel_size, filters = 48, 4, 32

    indices = create_dummy_indices(output_points, input_points, kernel_size)
    conv = SpatialConv(output_points=output_points, connection_indices=indices, filters=filters)
    x = torch.randn(batch_size, input_points, input_channels)
    y = conv(x)

    expected_shape = (batch_size, output_points, filters)
    actual_shape = tuple(y.shape)
    passed = actual_shape == expected_shape
    return passed, f"Expected: {expected_shape}, Got: {actual_shape}"


def test_spatial_mlp_shape() -> Tuple[bool, str]:
    """Test SpatialMLP output shape."""
    import torch
    from idx_flow import SpatialMLP

    batch_size, input_points, input_channels = 2, 192, 16
    output_points, kernel_size = 48, 4
    hidden_units = (32, 64, 32)

    indices = create_dummy_indices(output_points, input_points, kernel_size)
    mlp = SpatialMLP(
        output_points=output_points,
        connection_indices=indices,
        hidden_units=hidden_units,
        activations=("gelu", "gelu", "linear"),
    )
    x = torch.randn(batch_size, input_points, input_channels)
    y = mlp(x)

    expected_shape = (batch_size, output_points, hidden_units[-1])
    actual_shape = tuple(y.shape)
    passed = actual_shape == expected_shape
    return passed, f"Expected: {expected_shape}, Got: {actual_shape}"


def test_spatial_upsampling_shape() -> Tuple[bool, str]:
    """Test SpatialUpsampling output shape."""
    import torch
    from idx_flow import SpatialUpsampling

    batch_size, input_points, input_channels = 2, 48, 16
    output_points, kernel_size = 192, 4

    indices = create_dummy_indices(output_points, input_points, kernel_size)
    distances = create_dummy_distances(output_points, kernel_size)
    upsample = SpatialUpsampling(
        output_points=output_points,
        connection_indices=indices,
        distances=distances,
        interpolation="idw",
    )
    x = torch.randn(batch_size, input_points, input_channels)
    y = upsample(x)

    expected_shape = (batch_size, output_points, input_channels)
    actual_shape = tuple(y.shape)
    passed = actual_shape == expected_shape
    return passed, f"Expected: {expected_shape}, Got: {actual_shape}"


def test_global_mlp_shape() -> Tuple[bool, str]:
    """Test GlobalMLP output shape."""
    import torch
    from idx_flow import GlobalMLP

    batch_size, num_points, input_channels = 2, 192, 16
    hidden_units = (32, 64, 48)

    mlp = GlobalMLP(
        hidden_units=hidden_units,
        activations=("gelu", "gelu", "linear"),
        dropout=0.1,
        residual=False,
    )
    x = torch.randn(batch_size, num_points, input_channels)
    y = mlp(x)

    expected_shape = (batch_size, num_points, hidden_units[-1])
    actual_shape = tuple(y.shape)
    passed = actual_shape == expected_shape
    return passed, f"Expected: {expected_shape}, Got: {actual_shape}"


def test_squeeze_unsqueeze_shape() -> Tuple[bool, str]:
    """Test Squeeze and Unsqueeze shapes."""
    import torch
    from idx_flow import Squeeze, Unsqueeze

    batch_size, num_points, channels = 2, 192, 32

    x = torch.randn(batch_size, num_points, channels)
    squeeze = Squeeze(reduction="mean")
    squeezed = squeeze(x)

    unsqueeze = Unsqueeze(num_points=num_points)
    unsqueezed = unsqueeze(squeezed)

    squeeze_ok = tuple(squeezed.shape) == (batch_size, channels)
    unsqueeze_ok = tuple(unsqueezed.shape) == (batch_size, num_points, channels)
    passed = squeeze_ok and unsqueeze_ok
    return passed, f"Squeeze: {squeeze_ok}, Unsqueeze: {unsqueeze_ok}"


def test_attention_shape() -> Tuple[bool, str]:
    """Test SpatialSelfAttention shape."""
    import torch
    from idx_flow import SpatialSelfAttention

    batch_size, num_points, embed_dim = 2, 64, 32
    num_heads = 4

    attn = SpatialSelfAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=0.1)
    x = torch.randn(batch_size, num_points, embed_dim)
    y = attn(x)

    expected_shape = (batch_size, num_points, embed_dim)
    actual_shape = tuple(y.shape)
    passed = actual_shape == expected_shape
    return passed, f"Expected: {expected_shape}, Got: {actual_shape}"


# =============================================================================
# Test 3: Gradient Verification
# =============================================================================


def test_spatial_conv_gradients() -> Tuple[bool, str]:
    """Test gradient flow through SpatialConv."""
    import torch
    from idx_flow import SpatialConv

    batch_size, input_points, input_channels = 2, 192, 16
    output_points, kernel_size, filters = 48, 4, 32

    indices = create_dummy_indices(output_points, input_points, kernel_size)
    conv = SpatialConv(output_points=output_points, connection_indices=indices, filters=filters)
    x = torch.randn(batch_size, input_points, input_channels, requires_grad=True)
    y = conv(x)
    loss = (y ** 2).mean()
    loss.backward()

    input_grad_ok = x.grad is not None and not torch.isnan(x.grad).any()
    kernel_grad_ok = conv.kernel is not None and conv.kernel.grad is not None
    passed = input_grad_ok and kernel_grad_ok
    return passed, f"Input grad: {input_grad_ok}, Kernel grad: {kernel_grad_ok}"


def test_spatial_mlp_gradients() -> Tuple[bool, str]:
    """Test gradient flow through SpatialMLP with new features."""
    import torch
    from idx_flow import SpatialMLP

    batch_size, input_points, input_channels = 2, 192, 16
    output_points, kernel_size = 48, 4

    indices = create_dummy_indices(output_points, input_points, kernel_size)
    mlp = SpatialMLP(
        output_points=output_points,
        connection_indices=indices,
        hidden_units=(32, 32),
        activations=("gelu", "linear"),
        dropout=0.1,
        use_batch_norm=True,
    )
    mlp.train()
    x = torch.randn(batch_size, input_points, input_channels, requires_grad=True)
    y = mlp(x)
    loss = (y ** 2).mean()
    loss.backward()

    input_grad_ok = x.grad is not None and not torch.isnan(x.grad).any()
    mlp_grads_ok = all(
        layer.weight.grad is not None for layer in mlp.mlp_layers
    )
    passed = input_grad_ok and mlp_grads_ok
    return passed, f"Input grad: {input_grad_ok}, MLP grads: {mlp_grads_ok}"


def test_global_mlp_gradients() -> Tuple[bool, str]:
    """Test gradient flow through GlobalMLP."""
    import torch
    from idx_flow import GlobalMLP

    batch_size, num_points, input_channels = 2, 192, 16

    mlp = GlobalMLP(
        hidden_units=(32, 32),
        activations=("gelu", "linear"),
        dropout=0.1,
        residual=True,
    )
    mlp.train()
    x = torch.randn(batch_size, num_points, input_channels, requires_grad=True)
    y = mlp(x)
    loss = (y ** 2).mean()
    loss.backward()

    input_grad_ok = x.grad is not None and not torch.isnan(x.grad).any()
    mlp_grads_ok = all(
        layer.weight.grad is not None for layer in mlp.mlp_layers
    )
    passed = input_grad_ok and mlp_grads_ok
    return passed, f"Input grad: {input_grad_ok}, MLP grads: {mlp_grads_ok}"


def test_attention_gradients() -> Tuple[bool, str]:
    """Test gradient flow through SpatialSelfAttention."""
    import torch
    from idx_flow import SpatialSelfAttention

    batch_size, num_points, embed_dim = 2, 64, 32
    num_heads = 4

    attn = SpatialSelfAttention(embed_dim=embed_dim, num_heads=num_heads)
    x = torch.randn(batch_size, num_points, embed_dim, requires_grad=True)
    y = attn(x)
    loss = (y ** 2).mean()
    loss.backward()

    input_grad_ok = x.grad is not None and not torch.isnan(x.grad).any()
    qkv_grad_ok = attn.qkv_proj.weight.grad is not None
    passed = input_grad_ok and qkv_grad_ok
    return passed, f"Input grad: {input_grad_ok}, QKV grad: {qkv_grad_ok}"


# =============================================================================
# Test 4: New Features Verification
# =============================================================================


def test_initialization_options() -> Tuple[bool, str]:
    """Test different weight initialization options."""
    import torch
    from idx_flow import SpatialConv

    batch_size, input_points, input_channels = 2, 192, 16
    output_points, kernel_size, filters = 48, 4, 32
    indices = create_dummy_indices(output_points, input_points, kernel_size)

    init_methods = ["xavier_uniform", "xavier_normal", "kaiming_uniform",
                    "kaiming_normal", "orthogonal", "normal"]

    for method in init_methods:
        conv = SpatialConv(
            output_points=output_points,
            connection_indices=indices,
            filters=filters,
            weight_init=method,
        )
        x = torch.randn(batch_size, input_points, input_channels)
        y = conv(x)
        if torch.isnan(y).any():
            return False, f"NaN in output with {method} initialization"

    return True, f"All init methods work: {init_methods}"


def test_activation_options() -> Tuple[bool, str]:
    """Test different activation function options."""
    import torch
    from idx_flow import get_activation

    activations = ["relu", "selu", "leaky_relu", "gelu", "elu", "tanh",
                   "sigmoid", "swish", "mish", "linear"]

    x = torch.randn(2, 100)
    for act_name in activations:
        act = get_activation(act_name)
        y = act(x)
        if torch.isnan(y).any():
            return False, f"NaN in output with {act_name} activation"

    return True, f"All activations work: {activations}"


def test_normalization_layers() -> Tuple[bool, str]:
    """Test all normalization layer types."""
    import torch
    from idx_flow import SpatialBatchNorm, SpatialLayerNorm, SpatialInstanceNorm, SpatialGroupNorm

    batch_size, num_points, channels = 4, 192, 64
    x = torch.randn(batch_size, num_points, channels)

    # BatchNorm
    bn = SpatialBatchNorm(channels)
    bn.train()
    y_bn = bn(x)
    bn_ok = tuple(y_bn.shape) == (batch_size, num_points, channels)

    # LayerNorm
    ln = SpatialLayerNorm(channels)
    y_ln = ln(x)
    ln_ok = tuple(y_ln.shape) == (batch_size, num_points, channels)

    # InstanceNorm
    instn = SpatialInstanceNorm(channels)
    y_in = instn(x)
    in_ok = tuple(y_in.shape) == (batch_size, num_points, channels)

    # GroupNorm
    gn = SpatialGroupNorm(num_groups=8, num_channels=channels)
    y_gn = gn(x)
    gn_ok = tuple(y_gn.shape) == (batch_size, num_points, channels)

    passed = bn_ok and ln_ok and in_ok and gn_ok
    return passed, f"BN: {bn_ok}, LN: {ln_ok}, IN: {in_ok}, GN: {gn_ok}"


def test_dropout_layers() -> Tuple[bool, str]:
    """Test dropout layers."""
    import torch
    from idx_flow import SpatialDropout, ChannelDropout

    batch_size, num_points, channels = 4, 192, 64
    x = torch.randn(batch_size, num_points, channels)

    # SpatialDropout
    sd = SpatialDropout(p=0.5)
    sd.train()
    y_sd = sd(x)
    sd_ok = tuple(y_sd.shape) == (batch_size, num_points, channels)
    # Check some values are zeroed during training
    sd_drops = (y_sd == 0).any().item()

    # ChannelDropout
    cd = ChannelDropout(p=0.5)
    cd.train()
    y_cd = cd(x)
    cd_ok = tuple(y_cd.shape) == (batch_size, num_points, channels)

    # In eval mode, dropout should be identity
    sd.eval()
    y_sd_eval = sd(x)
    eval_ok = torch.allclose(x, y_sd_eval)

    passed = sd_ok and cd_ok and eval_ok
    return passed, f"SpatialDropout: {sd_ok}, ChannelDropout: {cd_ok}, Eval mode: {eval_ok}"


def test_mlp_with_residual() -> Tuple[bool, str]:
    """Test SpatialMLP with residual connections."""
    import torch
    from idx_flow import SpatialMLP

    batch_size, input_points, input_channels = 2, 192, 16
    output_points, kernel_size = 48, 4
    # Output dim matches kernel_size * input_channels for residual
    hidden_units = (32, 64)  # Output is 64

    indices = create_dummy_indices(output_points, input_points, kernel_size)

    mlp = SpatialMLP(
        output_points=output_points,
        connection_indices=indices,
        hidden_units=hidden_units,
        activations=("gelu", "linear"),
        residual=True,
        weight_init="xavier_uniform",
    )

    x = torch.randn(batch_size, input_points, input_channels, requires_grad=True)
    y = mlp(x)
    loss = (y ** 2).mean()
    loss.backward()

    shape_ok = tuple(y.shape) == (batch_size, output_points, hidden_units[-1])
    grad_ok = x.grad is not None and not torch.isnan(x.grad).any()
    passed = shape_ok and grad_ok
    return passed, f"Shape: {shape_ok}, Gradients: {grad_ok}"


# =============================================================================
# Main Execution
# =============================================================================


def main() -> int:
    """Run all verification tests and report results."""
    print("\n" + "#" * 60)
    print("#" + " " * 58 + "#")
    print("#" + "  idx-flow Package Verification".center(58) + "#")
    print("#" + "  Runtime Integrity Check (Extended)".center(58) + "#")
    print("#" + " " * 58 + "#")
    print("#" * 60)

    results: Dict[str, List[bool]] = {
        "imports": [],
        "shapes": [],
        "gradients": [],
        "new_features": [],
    }

    # Phase 1: Import Verification
    print_header("Phase 1: Import Verification")
    results["imports"].append(run_test("Import core layers", test_import_core_layers))
    results["imports"].append(run_test("Import MLP layers", test_import_mlp_layers))
    results["imports"].append(run_test("Import normalization layers", test_import_normalization_layers))
    results["imports"].append(run_test("Import regularization layers", test_import_regularization_layers))
    results["imports"].append(run_test("Import attention layers", test_import_attention_layers))
    results["imports"].append(run_test("Import utility layers", test_import_utility_layers))
    results["imports"].append(run_test("Import utility functions", test_import_utilities))

    # Phase 2: Shape Verification
    print_header("Phase 2: Shape Verification")
    results["shapes"].append(run_test("SpatialConv shape", test_spatial_conv_shape))
    results["shapes"].append(run_test("SpatialMLP shape", test_spatial_mlp_shape))
    results["shapes"].append(run_test("SpatialUpsampling shape", test_spatial_upsampling_shape))
    results["shapes"].append(run_test("GlobalMLP shape", test_global_mlp_shape))
    results["shapes"].append(run_test("Squeeze/Unsqueeze shape", test_squeeze_unsqueeze_shape))
    results["shapes"].append(run_test("SpatialSelfAttention shape", test_attention_shape))

    # Phase 3: Gradient Verification
    print_header("Phase 3: Gradient Verification")
    results["gradients"].append(run_test("SpatialConv gradients", test_spatial_conv_gradients))
    results["gradients"].append(run_test("SpatialMLP gradients", test_spatial_mlp_gradients))
    results["gradients"].append(run_test("GlobalMLP gradients", test_global_mlp_gradients))
    results["gradients"].append(run_test("SpatialSelfAttention gradients", test_attention_gradients))

    # Phase 4: New Features Verification
    print_header("Phase 4: New Features Verification")
    results["new_features"].append(run_test("Initialization options", test_initialization_options))
    results["new_features"].append(run_test("Activation options", test_activation_options))
    results["new_features"].append(run_test("Normalization layers", test_normalization_layers))
    results["new_features"].append(run_test("Dropout layers", test_dropout_layers))
    results["new_features"].append(run_test("MLP with residual", test_mlp_with_residual))

    # Final Summary
    print_header("FINAL SUMMARY")

    total_tests = sum(len(v) for v in results.values())
    total_passed = sum(sum(v) for v in results.values())

    for phase, phase_results in results.items():
        phase_passed = sum(phase_results)
        phase_total = len(phase_results)
        status = "PASS" if phase_passed == phase_total else "FAIL"
        color = "\033[92m" if status == "PASS" else "\033[91m"
        print(f"  {phase.upper():15} [{color}{status}\033[0m] {phase_passed}/{phase_total}")

    print("-" * 60)
    overall_status = "PASS" if total_passed == total_tests else "FAIL"
    overall_color = "\033[92m" if overall_status == "PASS" else "\033[91m"
    print(f"  {'OVERALL':15} [{overall_color}{overall_status}\033[0m] {total_passed}/{total_tests}")
    print()

    if total_passed == total_tests:
        print("  " + "\033[92m" + "BUILD VERIFICATION SUCCESSFUL" + "\033[0m")
        print("  All checks passed. Package is ready for deployment.")
        return 0
    else:
        print("  " + "\033[91m" + "BUILD VERIFICATION FAILED" + "\033[0m")
        print("  Some checks failed. Please review and fix before deployment.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
