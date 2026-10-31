"""Frequency-domain image filtering used by FreqWarm experiments.

This is a vectorized implementation of the official FreqWarm core operation:
apply a radial low-pass filter to RGB images before passing them through the
frozen autoencoder.
"""

import torch


def make_frequency_mask(
    height,
    width,
    cutoff=0.2,
    filter_type="ideal",
    order=2,
    *,
    device=None,
    dtype=torch.float32,
):
    """Create a radial low-pass mask in normalized cycles per pixel."""
    if cutoff <= 0:
        raise ValueError(f"cutoff must be positive, got {cutoff}")

    fy = torch.fft.fftshift(
        torch.fft.fftfreq(height, d=1.0, device=device)
    )
    fx = torch.fft.fftshift(
        torch.fft.fftfreq(width, d=1.0, device=device)
    )
    grid_y, grid_x = torch.meshgrid(fy, fx, indexing="ij")
    radius = torch.sqrt(grid_y.square() + grid_x.square())

    if filter_type == "ideal":
        mask = radius <= cutoff
    elif filter_type == "butterworth":
        mask = 1.0 / (1.0 + (radius / cutoff).pow(2 * order))
    elif filter_type == "gaussian":
        mask = torch.exp(-0.5 * (radius / cutoff).square())
    else:
        raise ValueError(f"Unknown filter_type: {filter_type}")

    return mask.to(dtype=dtype)


def apply_frequency_filter(
    images,
    *,
    keep="low",
    cutoff=0.2,
    filter_type="ideal",
    order=2,
):
    """Filter a ``(B, C, H, W)`` image batch along its spatial dimensions."""
    if images.ndim != 4:
        raise ValueError(
            f"images must have shape (B, C, H, W), got {tuple(images.shape)}"
        )
    if keep not in {"low", "high"}:
        raise ValueError(f"keep must be 'low' or 'high', got {keep!r}")

    height, width = images.shape[-2:]
    mask = make_frequency_mask(
        height,
        width,
        cutoff,
        filter_type,
        order,
        device=images.device,
        dtype=images.dtype,
    )
    if keep == "high":
        mask = 1.0 - mask

    spectrum = torch.fft.fftshift(
        torch.fft.fft2(images, dim=(-2, -1)),
        dim=(-2, -1),
    )
    filtered = torch.fft.ifft2(
        torch.fft.ifftshift(
            spectrum * mask[None, None],
            dim=(-2, -1),
        ),
        dim=(-2, -1),
    )
    return filtered.real
